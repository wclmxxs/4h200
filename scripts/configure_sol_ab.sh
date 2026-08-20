#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

mode=${1:-}
if [[ ${mode} != enable && ${mode} != disable ]]; then
  echo "Usage: $0 enable|disable" >&2
  exit 2
fi
if [[ ! -f .env ]]; then
  echo ".env is missing; run ./install.sh once before configuring Sol-Attn A/B" >&2
  exit 1
fi
mkdir -p .state .generated
sudo -v
exec 9>.state/install.lock
if ! flock -n 9; then
  echo "another install or attention reconfiguration is running" >&2
  exit 1
fi

set_env() {
  local key=$1 value=$2
  if grep -q "^${key}=" .env; then
    sed -i "s|^${key}=.*$|${key}=${value}|" .env
  else
    printf '%s=%s\n' "${key}" "${value}" >> .env
  fi
}

set_env_default() {
  local key=$1 value=$2
  if ! grep -q "^${key}=" .env; then
    printf '%s=%s\n' "${key}" "${value}" >> .env
  fi
}

set_env_default SOL_AB_SLOT 1
set_env_default SGLANG_SOL_IMAGE minimax-h3-h200-sglang-sol:20260820-v1
set_env_default SOL_ATTENTION_REVISION 5fe5febdf0f59fee1c0b44a5ce6665df0dabd247
set_env_default SOL_COMPONENT_ATTENTION_BACKENDS text_encoder=torch_sdpa,transformer=sol_attn
set_env_default SOL_ATTENTION_BACKEND_CONFIG dense_backend=sage_attn,dense_steps=2,kv_splits=auto,tau=1.0
set_env_default SOL_ATTN_STRICT 1
set_env_default SOL_WARMUP_STEPS 3

if [[ ${mode} == enable ]]; then
  set_env SOL_AB_ENABLED 1
else
  set_env SOL_AB_ENABLED 0
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT:-${DATA_ROOT}/hf-cache}
gpu_count=$(nvidia-smi -L | sed -n 's/^GPU [0-9][0-9]*:.*/x/p' | wc -l | tr -d ' ')
service_count=$((gpu_count / 4))
if (( service_count < 2 )); then
  echo "Sol-Attn A/B requires at least 8 GPUs; detected ${gpu_count}" >&2
  exit 1
fi
if (( SOL_AB_SLOT <= 0 || SOL_AB_SLOT >= service_count )); then
  echo "SOL_AB_SLOT must select a non-zero 4-GPU group; got ${SOL_AB_SLOT} for ${service_count} groups" >&2
  exit 1
fi

if ! sudo docker image inspect "${SGLANG_IMAGE}" >/dev/null 2>&1; then
  echo "baseline image ${SGLANG_IMAGE} is missing; run ./install.sh first" >&2
  exit 1
fi

if [[ ${mode} == enable ]]; then
  if [[ ${FORCE_BUILD_SOL:-0} == 1 ]] \
    || ! sudo docker image inspect "${SGLANG_SOL_IMAGE}" >/dev/null 2>&1; then
    echo "Building pinned Sol-Attn overlay image ${SGLANG_SOL_IMAGE}..."
    sudo docker build --progress=plain \
      --build-arg "SGLANG_BASE_IMAGE=${SGLANG_IMAGE}" \
      --build-arg "SOL_ATTENTION_REVISION=${SOL_ATTENTION_REVISION}" \
      -f docker/Dockerfile.sol-attn -t "${SGLANG_SOL_IMAGE}" .
  else
    echo "Reusing Sol-Attn image ${SGLANG_SOL_IMAGE}; set FORCE_BUILD_SOL=1 to rebuild it"
  fi
fi

generate_args=(
  --data-root "${DATA_ROOT}"
  --model-cache-root "${MODEL_CACHE_ROOT}"
  --advertise-host "${ADVERTISE_HOST}"
  --instance-id "${INSTANCE_ID}"
  --base-port "${API_BASE_PORT}"
  --release-id "${RELEASE_ID}"
  --sglang-image "${SGLANG_IMAGE}"
  --sglang-sol-image "${SGLANG_SOL_IMAGE}"
  --sol-ab-slot "${SOL_AB_SLOT}"
  --sol-component-attention-backends "${SOL_COMPONENT_ATTENTION_BACKENDS}"
  --sol-attention-backend-config "${SOL_ATTENTION_BACKEND_CONFIG}"
  --api-image "${API_IMAGE}"
)
if [[ ${mode} == enable ]]; then
  generate_args+=(--sol-ab-enabled)
fi
if [[ ${ALLOW_NON_H200:-0} == 1 ]]; then
  generate_args+=(--allow-non-h200)
fi
python3 scripts/generate_compose.py "${generate_args[@]}"

compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)
worker_service="h3-sglang-${SOL_AB_SLOT}"
worker_container="minimax-h3-h200-sglang-${SOL_AB_SLOT}"
echo "Restarting only partition ${SOL_AB_SLOT} (GPUs $((SOL_AB_SLOT * 4))-$((SOL_AB_SLOT * 4 + 3)))..."
"${compose[@]}" up -d --no-deps --force-recreate "${worker_service}"

startup_timeout=${STARTUP_TIMEOUT_SECONDS:-1800}
progress_interval=${STARTUP_PROGRESS_SECONDS:-15}
deadline=$((SECONDS + startup_timeout))
next_report=$SECONDS
initial_restarts=$(sudo docker inspect -f '{{.RestartCount}}' "${worker_container}" 2>/dev/null || echo 0)
while (( SECONDS < deadline )); do
  snapshot=$(sudo docker inspect \
    -f '{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|{{.RestartCount}}' \
    "${worker_container}" 2>/dev/null || true)
  IFS='|' read -r state health current_restarts <<<"${snapshot:-missing|none|0}"
  if [[ ${health} == healthy ]]; then
    break
  fi
  if [[ ${state} == exited || ${state} == dead || ${health} == unhealthy ]] \
    || (( current_restarts >= initial_restarts + 3 )); then
    sudo docker logs --tail 300 "${worker_container}" >&2 || true
    echo "partition ${SOL_AB_SLOT} failed: state=${state} health=${health} restarts=${current_restarts}" >&2
    exit 1
  fi
  if (( SECONDS >= next_report )); then
    echo "Waiting for partition ${SOL_AB_SLOT}: state=${state} health=${health} restarts=${current_restarts}"
    sudo docker logs --tail 2 "${worker_container}" 2>&1 \
      | sed "s/^/[partition ${SOL_AB_SLOT}] /" || true
    next_report=$((SECONDS + progress_interval))
  fi
  sleep 5
done
if [[ ${health:-none} != healthy ]]; then
  sudo docker logs --tail 300 "${worker_container}" >&2 || true
  echo "partition ${SOL_AB_SLOT} timed out after ${startup_timeout}s" >&2
  exit 1
fi

api_service="h3-api-${SOL_AB_SLOT}"
api_port=$((API_BASE_PORT + SOL_AB_SLOT))
echo "Refreshing API metadata for partition ${SOL_AB_SLOT}..."
"${compose[@]}" up -d --no-deps --force-recreate "${api_service}"
api_deadline=$((SECONDS + 180))
while (( SECONDS < api_deadline )); do
  if curl -fsS "http://127.0.0.1:${api_port}/healthz" \
    -H "Authorization: Bearer ${API_KEY}" \
    | jq -e '.ok == true' >/dev/null 2>&1; then
    break
  fi
  sleep 3
done
if (( SECONDS >= api_deadline )); then
  "${compose[@]}" logs --tail 200 "${api_service}" >&2 || true
  echo "API partition ${SOL_AB_SLOT} did not become healthy" >&2
  exit 1
fi

actual_image=$(sudo docker inspect -f '{{.Config.Image}}' "${worker_container}")
if [[ ${mode} == enable ]]; then
  [[ ${actual_image} == "${SGLANG_SOL_IMAGE}" ]] || {
    echo "expected ${SGLANG_SOL_IMAGE}, got ${actual_image}" >&2
    exit 1
  }
  sudo docker exec "${worker_container}" python3 -c 'import sol_attn; print("Sol-Attn import OK:", sol_attn.__file__)'
  worker_env=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "${worker_container}")
  grep -Fx 'ATTENTION_BACKEND=fa' <<<"${worker_env}"
  grep -Fx "COMPONENT_ATTENTION_BACKENDS=${SOL_COMPONENT_ATTENTION_BACKENDS}" <<<"${worker_env}"
  grep -Fx "ATTENTION_BACKEND_CONFIG=${SOL_ATTENTION_BACKEND_CONFIG}" <<<"${worker_env}"
  grep -Fx "SOL_ATTN_STRICT=${SOL_ATTN_STRICT}" <<<"${worker_env}"
  grep -Fx "WARMUP_STEPS=${SOL_WARMUP_STEPS}" <<<"${worker_env}"
  if ! sudo docker logs "${worker_container}" 2>&1 \
    | grep -Fq 'Attention backends for transformer: sol_attn (component constraint)'; then
    echo "worker became healthy but did not log the transformer sol_attn component constraint" >&2
    exit 1
  fi
  echo "SOL_AB_READY: port ${API_BASE_PORT}=Sage baseline; port $((API_BASE_PORT + SOL_AB_SLOT))=Sol-Attn"
else
  [[ ${actual_image} == "${SGLANG_IMAGE}" ]] || {
    echo "expected ${SGLANG_IMAGE}, got ${actual_image}" >&2
    exit 1
  }
  echo "SOL_AB_DISABLED: partition ${SOL_AB_SLOT} is back on Sage baseline image ${SGLANG_IMAGE}"
fi
