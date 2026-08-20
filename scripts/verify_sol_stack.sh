#!/usr/bin/env bash
set -euo pipefail

container=${1:?"Usage: $0 CONTAINER"}

required_env() {
  local key=$1 expected=$2
  if ! grep -Fx "${key}=${expected}" <<<"${worker_env}" >/dev/null; then
    echo "${container}: expected ${key}=${expected}" >&2
    exit 1
  fi
}

worker_env=$(sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "${container}")
required_env ATTENTION_BACKEND sol_attn
required_env COMPONENT_ATTENTION_BACKENDS "${SOL_COMPONENT_ATTENTION_BACKENDS}"
required_env ATTENTION_BACKEND_CONFIG "${SOL_ATTENTION_BACKEND_CONFIG}"
required_env SOL_ATTN_STRICT "${SOL_ATTN_STRICT}"
required_env WARMUP_STEPS "${SOL_WARMUP_STEPS}"
required_env QUANTIZATION "${SOL_QUANTIZATION}"
required_env LORA_MERGE_MODE "${SOL_LORA_MERGE_MODE}"
required_env SGLANG_CACHE_DIT_ENABLED "${SOL_CACHE_DIT_ENABLED}"
required_env SGLANG_CACHE_DIT_FN "${SOL_CACHE_DIT_FN}"
required_env SGLANG_CACHE_DIT_BN "${SOL_CACHE_DIT_BN}"
required_env SGLANG_CACHE_DIT_WARMUP "${SOL_CACHE_DIT_WARMUP}"
required_env SGLANG_CACHE_DIT_RDT "${SOL_CACHE_DIT_RDT}"
required_env SGLANG_CACHE_DIT_MC "${SOL_CACHE_DIT_MC}"

sudo docker exec -i "${container}" python3 - <<'PY'
import cache_dit
import os
import sol_attn
from pathlib import Path
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

assert envs.SGLANG_CACHE_DIT_ENABLED is True
assert envs.SGLANG_CACHE_DIT_FN == int(os.environ["SGLANG_CACHE_DIT_FN"])
assert envs.SGLANG_CACHE_DIT_BN == int(os.environ["SGLANG_CACHE_DIT_BN"])
assert envs.SGLANG_CACHE_DIT_WARMUP == int(os.environ["SGLANG_CACHE_DIT_WARMUP"])
assert envs.SGLANG_CACHE_DIT_RDT == float(os.environ["SGLANG_CACHE_DIT_RDT"])
assert envs.SGLANG_CACHE_DIT_MC == int(os.environ["SGLANG_CACHE_DIT_MC"])
commands = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        commands.append((proc / "cmdline").read_bytes().replace(b"\0", b" ").decode())
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
assert any(
    "sglang serve" in command
    and f"--quantization {os.environ['QUANTIZATION']}" in command
    and f"--lora-merge-mode {os.environ['LORA_MERGE_MODE']}" in command
    for command in commands
), "live sglang process is missing the requested FP8 or LoRA mode"
print("optimization imports OK:", sol_attn.__file__, cache_dit.__file__, Fp8Config.get_name())
PY

worker_logs=$(sudo docker logs "${container}" 2>&1)
grep -Fq 'Using sol_attn attention backend' <<<"${worker_logs}" || {
  echo "${container}: MiniMax H3 did not resolve its lazy DiT backend to sol_attn" >&2
  exit 1
}
grep -Fq 'Attention backends for audio_vae: fa' <<<"${worker_logs}" || {
  echo "${container}: audio_vae was not explicitly isolated on fa" >&2
  exit 1
}

echo "OPTIMIZATION_STACK_VERIFIED: Sol-Attn + FP8 + Cache-DiT (${SOL_CACHE_DIT_WARMUP}/${SOL_CACHE_DIT_RDT}/${SOL_CACHE_DIT_MC})"
