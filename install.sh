#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"
mkdir -p .state .generated

from_ami=false
case ${1:-} in
  "") ;;
  --from-ami) from_ami=true ;;
  -h|--help)
    echo "Usage: $0 [--from-ami]"
    echo "  --from-ami  Reuse baked images/cache while refreshing identity and GPU warmup."
    exit 0
    ;;
  *)
    echo "unknown argument: $1" >&2
    echo "Usage: $0 [--from-ami]" >&2
    exit 2
    ;;
esac
if (( $# > 1 )); then
  echo "Usage: $0 [--from-ami]" >&2
  exit 2
fi

if [[ ! -f .env ]]; then
  cp config/env.example .env
fi
chmod 600 .env

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

migrate_env_default() {
  local key=$1 old_value=$2 new_value=$3 current
  current=$(sed -n "s/^${key}=//p" .env)
  if [[ ${current} == "${old_value}" ]]; then
    set_env "${key}" "${new_value}"
  fi
}

sudo -v
# A cloned AMI can auto-start the baked reporter with the source node's
# identity. Stop it before any host/bootstrap work, then again after Docker is
# started because bootstrap may restart the daemon.
sudo docker stop minimax-h3-h200-reporter >/dev/null 2>&1 || true
if [[ ${from_ami} == true ]]; then
  echo "AMI fast path: validating baked host runtime"
  for command in curl jq openssl python3 flock findmnt nvidia-smi docker systemctl; do
    command -v "${command}" >/dev/null 2>&1 || {
      echo "AMI fast path requires ${command}; run ./install.sh once to repair the host" >&2
      exit 1
    }
  done
  nvidia-smi -L >/dev/null
  sudo systemctl enable --now docker >/dev/null
  sudo docker info >/dev/null
  sudo docker compose version
else
  sudo scripts/bootstrap_host.sh
fi
exec 9>.state/install.lock
if ! flock -n 9; then
  echo "another install.sh process is running" >&2
  exit 1
fi
sudo docker stop minimax-h3-h200-reporter >/dev/null 2>&1 || true

if [[ -z $(sed -n 's/^API_KEY=//p' .env) ]]; then
  set_env API_KEY "$(openssl rand -hex 32)"
fi

detect_imds() {
  local path=$1 token
  token=$(curl -fsS --connect-timeout 1 -X PUT \
    http://169.254.169.254/latest/api/token \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)
  [[ -n ${token} ]] || return 1
  curl -fsS --connect-timeout 1 \
    -H "X-aws-ec2-metadata-token: ${token}" \
    "http://169.254.169.254/latest/meta-data/${path}" 2>/dev/null
}

require_public_ipv4() {
  python3 - "$1" <<'PY'
import ipaddress
import sys

try:
    address = ipaddress.ip_address(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if address.version == 4 and address.is_global else 1)
PY
}

configured_advertise_host=$(sed -n 's/^ADVERTISE_HOST=//p' .env)
detected_advertise_host=$(detect_imds public-ipv4 || true)
if [[ -n ${detected_advertise_host} ]]; then
  advertise_host=${detected_advertise_host}
  if [[ ${configured_advertise_host} != "${advertise_host}" ]]; then
    echo "Refreshing ADVERTISE_HOST from AWS IMDS: ${configured_advertise_host:-<empty>} -> ${advertise_host}"
  fi
  set_env ADVERTISE_HOST "${advertise_host}"
else
  advertise_host=${configured_advertise_host}
  [[ -n ${advertise_host} ]] || {
    echo "AWS IMDSv2 did not return public-ipv4; set ADVERTISE_HOST to this node's public IPv4" >&2
    exit 1
  }
fi
if ! require_public_ipv4 "${advertise_host}"; then
  echo "ADVERTISE_HOST must be a public IPv4 address; got ${advertise_host}" >&2
  exit 1
fi

configured_instance_id=$(sed -n 's/^INSTANCE_ID=//p' .env)
detected_instance_id=$(detect_imds instance-id || true)
if [[ -n ${detected_instance_id} ]]; then
  instance_id=${detected_instance_id}
  if [[ ${configured_instance_id} != "${instance_id}" ]]; then
    echo "Refreshing INSTANCE_ID from AWS IMDS: ${configured_instance_id:-<empty>} -> ${instance_id}"
  fi
  set_env INSTANCE_ID "${instance_id}"
else
  instance_id=${configured_instance_id:-$(hostname)}
  [[ -n ${instance_id} ]] || { echo "unable to determine INSTANCE_ID" >&2; exit 1; }
  set_env INSTANCE_ID "${instance_id}"
fi

set_env HOST_UID "$(id -u)"
set_env HOST_GID "$(id -g)"
set_env_default VIDEO_RETENTION_HOURS 12
set_env_default CLEANUP_INTERVAL_SECONDS 600
set_env_default MODEL_CACHE_ROOT ""
set_env_default SAGEATTENTION_REVISION d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4
set_env_default ATTENTION_BACKEND fa
set_env_default COMPONENT_ATTENTION_BACKENDS transformer=sage_attn
set_env_default ATTENTION_BACKEND_CONFIG ""
set_env_default OPTIMIZATION_STACK_ENABLED 1
set_env_default SGLANG_SOL_IMAGE minimax-h3-h200-sglang-sol:20260820-v1
set_env_default SOL_ATTENTION_REVISION 5fe5febdf0f59fee1c0b44a5ce6665df0dabd247
set_env_default SOL_COMPONENT_ATTENTION_BACKENDS text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn
set_env_default SOL_ATTENTION_BACKEND_CONFIG dense_backend=sage_attn,dense_steps=2,kv_splits=auto,tau=1.0
set_env_default SOL_ATTN_STRICT 1
set_env_default SOL_WARMUP_STEPS 3
set_env_default SOL_QUANTIZATION fp8
set_env_default SOL_LORA_MERGE_MODE dynamic
set_env_default SOL_CACHE_DIT_ENABLED true
set_env_default SOL_CACHE_DIT_FN 1
set_env_default SOL_CACHE_DIT_BN 0
set_env_default SOL_CACHE_DIT_WARMUP 2
set_env_default SOL_CACHE_DIT_RDT 0.04
set_env_default SOL_CACHE_DIT_MC 1
set_env_default LORA_SIZE 779849816
migrate_env_default ATTENTION_BACKEND sage_attn fa
migrate_env_default COMPONENT_ATTENTION_BACKENDS text_encoder=torch_sdpa transformer=sage_attn
migrate_env_default RELEASE_ID h3-4h200-20260819-v1 h3-4h200-20260820-v2
migrate_env_default SGLANG_IMAGE minimax-h3-h200-sglang:20260819-v1 minimax-h3-h200-sglang:20260820-v2
migrate_env_default API_IMAGE minimax-h3-h200-api:20260819-v1 minimax-h3-h200-api:20260820-v2
migrate_env_default REPORTER_IMAGE minimax-h3-h200-reporter:20260819-v1 minimax-h3-h200-reporter:20260820-v2
migrate_env_default RELEASE_ID h3-4h200-20260820-v2 h3-4h200-20260820-v3
migrate_env_default SGLANG_IMAGE minimax-h3-h200-sglang:20260820-v2 minimax-h3-h200-sglang:20260820-v3
migrate_env_default API_IMAGE minimax-h3-h200-api:20260820-v2 minimax-h3-h200-api:20260820-v3
migrate_env_default REPORTER_IMAGE minimax-h3-h200-reporter:20260820-v2 minimax-h3-h200-reporter:20260820-v3
migrate_env_default RELEASE_ID h3-4h200-20260820-v3 h3-4h200-20260820-v4
migrate_env_default SGLANG_IMAGE minimax-h3-h200-sglang:20260820-v3 minimax-h3-h200-sglang:20260820-v4
migrate_env_default API_IMAGE minimax-h3-h200-api:20260820-v3 minimax-h3-h200-api:20260820-v4
migrate_env_default REPORTER_IMAGE minimax-h3-h200-reporter:20260820-v3 minimax-h3-h200-reporter:20260820-v4
migrate_env_default RELEASE_ID h3-4h200-20260820-v4 h3-4h200-20260820-v5
migrate_env_default API_IMAGE minimax-h3-h200-api:20260820-v4 minimax-h3-h200-api:20260820-v5
migrate_env_default RELEASE_ID h3-4h200-20260820-v5 h3-4h200-20260820-v6
migrate_env_default API_IMAGE minimax-h3-h200-api:20260820-v5 minimax-h3-h200-api:20260820-v6
migrate_env_default RELEASE_ID h3-4h200-20260820-v6 h3-4h200-20260820-v7
migrate_env_default RELEASE_ID h3-4h200-20260820-v7 h3-4h200-20260820-v8
migrate_env_default RELEASE_ID h3-4h200-20260820-v8 h3-4h200-20260820-v9
migrate_env_default RELEASE_ID h3-4h200-20260820-v9 h3-4h200-20260820-v10
migrate_env_default SOL_COMPONENT_ATTENTION_BACKENDS text_encoder=torch_sdpa,transformer=sol_attn text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn
migrate_env_default RELEASE_ID h3-4h200-20260820-v10 h3-4h200-20260820-v11
migrate_env_default API_IMAGE minimax-h3-h200-api:20260820-v6 minimax-h3-h200-api:20260820-v7

set -a
# shellcheck disable=SC1091
source .env
set +a

MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT:-${DATA_ROOT}/hf-cache}
export MODEL_CACHE_ROOT

if [[ ${SERVICE_ID} != "Minimax-H3-AWS-H200" ]]; then
  echo "SERVICE_ID must be Minimax-H3-AWS-H200; got ${SERVICE_ID}" >&2
  exit 1
fi
if [[ ${ULYSSES} != "4" ]]; then
  echo "ULYSSES must be 4 for a 4-GPU service; got ${ULYSSES}" >&2
  exit 1
fi

gpu_count=$(nvidia-smi -L | sed -n 's/^GPU [0-9][0-9]*:.*/x/p' | wc -l | tr -d ' ')
if (( gpu_count < 4 )); then
  echo "at least 4 GPUs are required; detected ${gpu_count}" >&2
  exit 1
fi
service_count=$((gpu_count / 4))
ignored_gpu_count=$((gpu_count % 4))
if (( ignored_gpu_count > 0 )); then
  echo "warning: ${ignored_gpu_count} trailing GPU(s) will not be assigned" >&2
fi

echo "Detected ${gpu_count} GPUs: ${service_count} x 4-H200 service(s) on ${INSTANCE_ID} (${ADVERTISE_HOST})"

sudo mkdir -p "${MODEL_CACHE_ROOT}" "${DATA_ROOT}/reporter" "${DATA_ROOT}/cleaner"
for ((slot=0; slot<service_count; slot++)); do
  sudo mkdir -p "${DATA_ROOT}/slots/${slot}/output" "${DATA_ROOT}/slots/${slot}/api-data"
done
sudo chown "$(id -u):$(id -g)" "${DATA_ROOT}" "${MODEL_CACHE_ROOT}"
sudo chown -R "$(id -u):$(id -g)" \
  "${DATA_ROOT}/reporter" "${DATA_ROOT}/cleaner" "${DATA_ROOT}/slots"
if [[ ! -w ${MODEL_CACHE_ROOT} ]]; then
  echo "Model cache ownership differs from this user; repairing ${MODEL_CACHE_ROOT}"
  sudo chown -R "$(id -u):$(id -g)" "${MODEL_CACHE_ROOT}"
fi

if [[ ${from_ami} == true ]]; then
  cache_source=$(findmnt -n -o SOURCE -T "${MODEL_CACHE_ROOT}" 2>/dev/null || true)
  if [[ ${MODEL_CACHE_ROOT} == /opt/dlami/nvme/* || ${cache_source} == *ephemeral* ]]; then
    echo "WARNING: MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT} is on instance-store NVMe (${cache_source:-unknown})."
    echo "WARNING: standard EC2 AMIs do not preserve this cache; use snapshot-backed EBS for fast cloned starts."
  fi
  if [[ ! -d ${MODEL_CACHE_ROOT}/hub/models--MiniMaxAI--MiniMax-H3 ]]; then
    echo "AMI fast path: MiniMax-H3 base-model cache is missing; SGLang will download it during startup."
  fi
fi

MODEL_VENV_DIR=".state/model-venv"
MODEL_VENV_PYTHON="${MODEL_VENV_DIR}/bin/python"
MODEL_VENV_PIP="${MODEL_VENV_DIR}/bin/pip"
MODEL_REQUIREMENTS="model-requirements.txt"

if [[ ! -x ${MODEL_VENV_PYTHON} ]]; then
  python3 -m venv "${MODEL_VENV_DIR}"
fi

if ! "${MODEL_VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
  if ! "${MODEL_VENV_PYTHON}" -m ensurepip --upgrade; then
    python3 -m venv --clear "${MODEL_VENV_DIR}"
  fi
fi

if ! "${MODEL_VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
  echo "pip is unavailable in ${MODEL_VENV_DIR}; install python3-venv/python3-pip and rerun" >&2
  exit 1
fi

if ! "${MODEL_VENV_PYTHON}" - <<'PY'
import importlib.util
import sys

required_modules = ("huggingface_hub", "hf_xet")
missing = [mod for mod in required_modules if importlib.util.find_spec(mod) is None]
if missing:
    sys.exit(1)
sys.exit(0)
PY
then
  "${MODEL_VENV_PYTHON}" -m pip install --upgrade pip
  "${MODEL_VENV_PYTHON}" -m pip install -r "${MODEL_REQUIREMENTS}"
fi

lora_local_path=$(
  lora_args=(
    --cache-root "${MODEL_CACHE_ROOT}"
    --repo "${LORA_REPO}"
    --revision "${LORA_REVISION}"
    --filename "${LORA_WEIGHT}"
    --sha256 "${LORA_SHA256}"
    --size "${LORA_SIZE}"
  )
  if [[ ${from_ami} == true ]]; then
    lora_args+=(--trust-existing-size)
  fi
  HF_TOKEN=${HF_TOKEN:-} "${MODEL_VENV_PYTHON}" scripts/download_lora.py "${lora_args[@]}"
)
set_env LORA_LOCAL_PATH "${lora_local_path}"
export LORA_LOCAL_PATH="${lora_local_path}"

docker_cmd=(sudo docker)
build_image() {
  local dockerfile=$1 image=$2
  shift 2
  if [[ ${from_ami} == true ]] \
    && "${docker_cmd[@]}" image inspect "${image}" >/dev/null 2>&1; then
    echo "AMI fast path: reusing Docker image ${image}"
    return
  fi
  "${docker_cmd[@]}" build --progress=plain "$@" \
    -f "${dockerfile}" -t "${image}" .
}
build_image docker/Dockerfile.sglang "${SGLANG_IMAGE}" \
  --build-arg "SGLANG_BASE_IMAGE=${SGLANG_BASE_IMAGE}" \
  --build-arg "SAGEATTENTION_REVISION=${SAGEATTENTION_REVISION:-d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4}"
if [[ ${OPTIMIZATION_STACK_ENABLED:-1} == "1" ]]; then
  build_image docker/Dockerfile.sol-attn "${SGLANG_SOL_IMAGE}" \
    --build-arg "SGLANG_BASE_IMAGE=${SGLANG_IMAGE}" \
    --build-arg "SOL_ATTENTION_REVISION=${SOL_ATTENTION_REVISION}"
fi
build_image docker/Dockerfile.api "${API_IMAGE}"
build_image docker/Dockerfile.reporter "${REPORTER_IMAGE}"

generate_args=(
  --data-root "${DATA_ROOT}"
  --model-cache-root "${MODEL_CACHE_ROOT}"
  --advertise-host "${ADVERTISE_HOST}"
  --instance-id "${INSTANCE_ID}"
  --base-port "${API_BASE_PORT}"
  --release-id "${RELEASE_ID}"
  --sglang-image "${SGLANG_IMAGE}"
  --sglang-sol-image "${SGLANG_SOL_IMAGE}"
  --sol-component-attention-backends "${SOL_COMPONENT_ATTENTION_BACKENDS}"
  --sol-attention-backend-config "${SOL_ATTENTION_BACKEND_CONFIG}"
  --api-image "${API_IMAGE}"
)
if [[ ${OPTIMIZATION_STACK_ENABLED:-1} == "1" ]]; then
  generate_args+=(--optimization-stack-enabled)
fi
if [[ ${ALLOW_NON_H200:-0} == "1" ]]; then
  generate_args+=(--allow-non-h200)
fi
python3 scripts/generate_compose.py "${generate_args[@]}"

compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)
"${compose[@]}" stop h3-reporter >/dev/null 2>&1 || true
"${compose[@]}" up -d h3-cleaner

startup_timeout=${STARTUP_TIMEOUT_SECONDS:-1800}
progress_interval=${STARTUP_PROGRESS_SECONDS:-15}

wait_for_worker() {
  local slot=$1
  local container="minimax-h3-h200-sglang-${slot}"
  local deadline=$((SECONDS + startup_timeout))
  local next_report=$SECONDS
  local initial_restarts current_restarts state health snapshot

  initial_restarts=$(sudo docker inspect -f '{{.RestartCount}}' "${container}" 2>/dev/null || echo 0)
  while (( SECONDS < deadline )); do
    snapshot=$(sudo docker inspect \
      -f '{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}|{{.RestartCount}}' \
      "${container}" 2>/dev/null || true)
    IFS='|' read -r state health current_restarts <<<"${snapshot:-missing|none|0}"
    if [[ ${health} == healthy ]]; then
      echo "4-H200 partition ${slot} is healthy after $((startup_timeout - deadline + SECONDS))s"
      return 0
    fi
    if [[ ${state} == exited || ${state} == dead || ${health} == unhealthy ]] \
      || (( current_restarts >= initial_restarts + 3 )); then
      echo "4-H200 partition ${slot} failed: state=${state} health=${health} restarts=${current_restarts}" >&2
      return 1
    fi
    if (( SECONDS >= next_report )); then
      echo "Waiting for partition ${slot}: state=${state} health=${health} restarts=${current_restarts} elapsed=$((startup_timeout - deadline + SECONDS))s"
      sudo docker logs --tail 2 "${container}" 2>&1 \
        | sed "s/^/[partition ${slot}] /" || true
      next_report=$((SECONDS + progress_interval))
    fi
    sleep 5
  done
  echo "4-H200 partition ${slot} timed out after ${startup_timeout}s" >&2
  return 1
}

wait_for_api() {
  local slot=$1
  local port=$((API_BASE_PORT + slot))
  local deadline=$((SECONDS + 180))
  local next_report=$SECONDS
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/healthz" \
      -H "Authorization: Bearer ${API_KEY}" \
      | jq -e '.ok == true' >/dev/null 2>&1; then
      echo "API partition ${slot} is healthy on port ${port}"
      return 0
    fi
    if (( SECONDS >= next_report )); then
      echo "Waiting for API partition ${slot} on port ${port}..."
      next_report=$((SECONDS + progress_interval))
    fi
    sleep 3
  done
  return 1
}

worker_services=()
for ((slot=0; slot<service_count; slot++)); do
  worker_services+=("h3-sglang-${slot}")
done
echo "Starting ${service_count} x 4-H200 partitions in parallel..."
"${compose[@]}" up -d "${worker_services[@]}"

for ((slot=0; slot<service_count; slot++)); do
  if ! wait_for_worker "${slot}"; then
    "${compose[@]}" logs --tail 300 "h3-sglang-${slot}"
    echo "SGLang partition ${slot} did not become healthy" >&2
    exit 1
  fi
done

if [[ ${OPTIMIZATION_STACK_ENABLED:-1} == "1" ]]; then
  for ((slot=0; slot<service_count; slot++)); do
    bash scripts/verify_sol_stack.sh "minimax-h3-h200-sglang-${slot}"
  done
fi

api_services=()
for ((slot=0; slot<service_count; slot++)); do
  api_services+=("h3-api-${slot}")
done
"${compose[@]}" up -d "${api_services[@]}"

for ((slot=0; slot<service_count; slot++)); do
  if ! wait_for_api "${slot}"; then
    "${compose[@]}" logs --tail 200 "h3-api-${slot}"
    echo "API partition ${slot} did not become healthy" >&2
    exit 1
  fi
done

report_started_at=$(date +%s)
"${compose[@]}" up -d --remove-orphans h3-cleaner h3-reporter
deadline=$((SECONDS + 90))
catalog_success=false
while (( SECONDS < deadline )); do
  if [[ -f ${DATA_ROOT}/reporter/status.json ]] \
    && jq -e --argjson started "${report_started_at}" \
      '.catalog_success == true
       and .timestamp >= $started
       and .healthy_instances == .instance_count' \
      "${DATA_ROOT}/reporter/status.json" >/dev/null 2>&1; then
    catalog_success=true
    break
  fi
  sleep 2
done
if [[ ${catalog_success} != true ]]; then
  "${compose[@]}" logs --tail 150 h3-reporter
  echo "services are healthy, but ReportCatalog registration did not succeed" >&2
  exit 1
fi

echo "READY: ${service_count} x 4-H200 services"
echo "PSM=${PSM}"
echo "SERVICE_ID=${SERVICE_ID}"
echo "Public endpoints: http://${ADVERTISE_HOST}:${API_BASE_PORT}-$((API_BASE_PORT + service_count - 1))"
jq '{catalog_success,healthy_instances,instance_count,catalog_response}' \
  "${DATA_ROOT}/reporter/status.json"
