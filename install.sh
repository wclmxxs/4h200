#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${ROOT}"
mkdir -p .state .generated

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

sudo -v
sudo scripts/bootstrap_host.sh
exec 9>.state/install.lock
if ! flock -n 9; then
  echo "another install.sh process is running" >&2
  exit 1
fi

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

advertise_host=$(sed -n 's/^ADVERTISE_HOST=//p' .env)
if [[ -z ${advertise_host} ]]; then
  advertise_host=$(detect_imds public-ipv4 || true)
  [[ -n ${advertise_host} ]] || {
    echo "AWS IMDSv2 did not return public-ipv4; set ADVERTISE_HOST to this node's public IPv4" >&2
    exit 1
  }
  set_env ADVERTISE_HOST "${advertise_host}"
fi
if ! require_public_ipv4 "${advertise_host}"; then
  echo "ADVERTISE_HOST must be a public IPv4 address; got ${advertise_host}" >&2
  exit 1
fi

instance_id=$(sed -n 's/^INSTANCE_ID=//p' .env)
if [[ -z ${instance_id} ]]; then
  instance_id=$(detect_imds instance-id || hostname)
  [[ -n ${instance_id} ]] || { echo "unable to determine INSTANCE_ID" >&2; exit 1; }
  set_env INSTANCE_ID "${instance_id}"
fi

set_env HOST_UID "$(id -u)"
set_env HOST_GID "$(id -g)"

set -a
# shellcheck disable=SC1091
source .env
set +a

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

sudo mkdir -p "${DATA_ROOT}/hf-cache" "${DATA_ROOT}/reporter"
for ((slot=0; slot<service_count; slot++)); do
  sudo mkdir -p "${DATA_ROOT}/slots/${slot}/output" "${DATA_ROOT}/slots/${slot}/api-data"
done
sudo chown -R "$(id -u):$(id -g)" "${DATA_ROOT}"

if [[ ! -x .state/model-venv/bin/python ]]; then
  python3 -m venv .state/model-venv
  .state/model-venv/bin/pip install --upgrade pip
  .state/model-venv/bin/pip install 'huggingface_hub>=0.34,<2' 'hf_xet>=1.1,<2'
fi
lora_local_path=$(
  HF_TOKEN=${HF_TOKEN:-} .state/model-venv/bin/python scripts/download_lora.py \
    --cache-root "${DATA_ROOT}/hf-cache" \
    --repo "${LORA_REPO}" \
    --revision "${LORA_REVISION}" \
    --filename "${LORA_WEIGHT}" \
    --sha256 "${LORA_SHA256}"
)
set_env LORA_LOCAL_PATH "${lora_local_path}"
export LORA_LOCAL_PATH="${lora_local_path}"

docker_cmd=(sudo docker)
"${docker_cmd[@]}" build --progress=plain \
  --build-arg "SGLANG_BASE_IMAGE=${SGLANG_BASE_IMAGE}" \
  -f docker/Dockerfile.sglang -t "${SGLANG_IMAGE}" .
"${docker_cmd[@]}" build --progress=plain \
  -f docker/Dockerfile.api -t "${API_IMAGE}" .
"${docker_cmd[@]}" build --progress=plain \
  -f docker/Dockerfile.reporter -t "${REPORTER_IMAGE}" .

generate_args=(
  --data-root "${DATA_ROOT}"
  --advertise-host "${ADVERTISE_HOST}"
  --instance-id "${INSTANCE_ID}"
  --base-port "${API_BASE_PORT}"
  --release-id "${RELEASE_ID}"
  --sglang-image "${SGLANG_IMAGE}"
  --api-image "${API_IMAGE}"
)
if [[ ${ALLOW_NON_H200:-0} == "1" ]]; then
  generate_args+=(--allow-non-h200)
fi
python3 scripts/generate_compose.py "${generate_args[@]}"

compose=(sudo docker compose --env-file .env -f .generated/compose.yaml)
"${compose[@]}" stop h3-reporter >/dev/null 2>&1 || true

startup_timeout=${STARTUP_TIMEOUT_SECONDS:-1800}
for ((slot=0; slot<service_count; slot++)); do
  echo "Starting 4-H200 partition ${slot}..."
  "${compose[@]}" up -d "h3-sglang-${slot}"
  deadline=$((SECONDS + startup_timeout))
  worker_healthy=false
  while (( SECONDS < deadline )); do
    health=$(sudo docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' \
      "minimax-h3-h200-sglang-${slot}" 2>/dev/null || true)
    if [[ ${health} == healthy ]]; then
      worker_healthy=true
      break
    fi
    if [[ ${health} == unhealthy ]]; then
      break
    fi
    sleep 5
  done
  if [[ ${worker_healthy} != true ]]; then
    "${compose[@]}" logs --tail 300 "h3-sglang-${slot}"
    echo "SGLang partition ${slot} did not become healthy" >&2
    exit 1
  fi

  "${compose[@]}" up -d "h3-api-${slot}"
  port=$((API_BASE_PORT + slot))
  api_healthy=false
  deadline=$((SECONDS + 180))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${port}/healthz" \
      -H "Authorization: Bearer ${API_KEY}" | jq -e '.ok == true' >/dev/null 2>&1; then
      api_healthy=true
      break
    fi
    sleep 3
  done
  if [[ ${api_healthy} != true ]]; then
    "${compose[@]}" logs --tail 200 "h3-api-${slot}"
    echo "API partition ${slot} did not become healthy" >&2
    exit 1
  fi
done

report_started_at=$(date +%s)
"${compose[@]}" up -d --remove-orphans h3-reporter
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
