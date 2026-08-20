#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

GPUS_PER_SERVICE = 4


def quote(value: object) -> str:
    return json.dumps(str(value))


def detect_gpus() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    gpus: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, uuid, name, memory = [part.strip() for part in line.split(",", 3)]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_mb": int(memory),
            }
        )
    if not gpus:
        raise SystemExit("nvidia-smi returned no GPUs")
    gpus.sort(key=lambda gpu: int(gpu["index"]))
    indexes = [gpu["index"] for gpu in gpus]
    if indexes != list(range(len(indexes))):
        raise SystemExit(f"GPU indexes must be contiguous from zero; got {indexes}")
    return gpus


def partition_gpus(
    gpus: list[dict[str, object]], allow_non_h200: bool = False
) -> list[list[dict[str, object]]]:
    if len(gpus) < GPUS_PER_SERVICE:
        raise SystemExit(
            f"at least {GPUS_PER_SERVICE} GPUs are required; got {len(gpus)}"
        )
    usable = len(gpus) - len(gpus) % GPUS_PER_SERVICE
    if not allow_non_h200:
        invalid = [
            gpu for gpu in gpus[:usable] if "H200" not in str(gpu["name"]).upper()
        ]
        if invalid:
            names = sorted({str(gpu["name"]) for gpu in invalid})
            raise SystemExit(f"all assigned GPUs must be H200; got {names}")
    if usable != len(gpus):
        ignored = [gpu["index"] for gpu in gpus[usable:]]
        print(
            f"warning: ignoring GPUs {ignored}; services require complete 4-GPU groups",
            file=sys.stderr,
        )
    return [gpus[index : index + GPUS_PER_SERVICE] for index in range(0, usable, 4)]


def detect_host_resources(group_count: int) -> tuple[int, int]:
    cpu_total = os.cpu_count() or group_count * 16
    memory_total_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        with meminfo.open() as source:
            for line in source:
                if line.startswith("MemTotal:"):
                    memory_total_mb = int(line.split()[1]) // 1024
                    break
    if memory_total_mb <= 0:
        memory_total_mb = group_count * 64 * 1024
    return max(4, cpu_total // group_count), max(16384, memory_total_mb // group_count)


def sglang_command() -> str:
    return """set -euo pipefail
cd /sgl-workspace/sglang
lora_path="$$LORA_REPO"
if [[ -n "$$LORA_LOCAL_PATH" ]]; then
  lora_path="$$LORA_LOCAL_PATH"
fi
args=(
  sglang serve
  --model-path "$$MODEL"
  --model-variant fl2va
  --num-gpus 4
  --tp-size "$$TP"
  --ulysses-degree "$$ULYSSES"
  --performance-mode speed
  --encoder-parallel "$$ENCODER_PARALLEL"
  --lora-path "$$lora_path"
  --lora-weight-name "$$LORA_WEIGHT"
  --lora-nickname "$$LORA_NICKNAME"
  --lora-scale "$$LORA_SCALE"
  --lora-merge-mode "$$LORA_MERGE_MODE"
  --output-path /out/videos
  --host 0.0.0.0
  --port 30020
)
if [[ -n "$$WARMUP" ]]; then
  read -r -a warmup <<<"$$WARMUP"
  args+=(--warmup-resolutions "$${warmup[@]}")
fi
exec "$${args[@]}"
"""


def sglang_service(
    group_index: int, group: list[dict[str, object]], data_root: str
) -> list[str]:
    indexes = [int(gpu["index"]) for gpu in group]
    slot = f"{data_root}/slots/{group_index}"
    device_ids = ", ".join(quote(index) for index in indexes)
    return [
        f"  h3-sglang-{group_index}:",
        "    image: ${SGLANG_IMAGE}",
        f"    container_name: minimax-h3-h200-sglang-{group_index}",
        "    restart: unless-stopped",
        "    init: true",
        "    ipc: host",
        "    shm_size: 32gb",
        "    env_file: ../.env",
        f'    command: ["bash", "-lc", {quote(sglang_command())}]',
        "    environment:",
        "      HF_HOME: /cache/huggingface",
        "      HF_HUB_CACHE: /cache/huggingface/hub",
        "      SGLANG_MINIMAX_H3_EXTRA_SHORT_EDGES: ${SHORT_EDGES:-480,704}",
        "    volumes:",
        f"      - {data_root}/hf-cache:/cache/huggingface",
        f"      - {slot}/output:/out/videos",
        "    healthcheck:",
        "      test: ['CMD-SHELL', 'curl -fsS http://127.0.0.1:30020/health >/dev/null']",
        "      interval: 10s",
        "      timeout: 5s",
        "      retries: 90",
        "      start_period: 120s",
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        f"              device_ids: [{device_ids}]",
        "              capabilities: [gpu]",
    ]


def api_service(
    group_index: int,
    group: list[dict[str, object]],
    data_root: str,
    host: str,
    base_port: int,
) -> list[str]:
    port = base_port + group_index
    slot = f"{data_root}/slots/{group_index}"
    indexes = ",".join(str(gpu["index"]) for gpu in group)
    uuids = ",".join(str(gpu["uuid"]) for gpu in group)
    return [
        f"  h3-api-{group_index}:",
        "    image: ${API_IMAGE}",
        f"    container_name: minimax-h3-h200-api-{group_index}",
        "    restart: unless-stopped",
        "    init: true",
        "    user: ${HOST_UID}:${HOST_GID}",
        "    env_file: ../.env",
        "    depends_on:",
        f"      h3-sglang-{group_index}:",
        "        condition: service_healthy",
        "    ports:",
        f"      - '0.0.0.0:{port}:30010'",
        "    environment:",
        f"      SGLANG_URL: http://h3-sglang-{group_index}:30020",
        "      DATA_ROOT: /data",
        f"      PUBLIC_BASE_URL: {quote(f'http://{host}:{port}')}",
        f"      GPU_GROUP_INDEX: {quote(group_index)}",
        f"      GPU_INDEXES: {quote(indexes)}",
        f"      GPU_UUIDS: {quote(uuids)}",
        "    volumes:",
        f"      - {slot}/api-data:/data",
        "    healthcheck:",
        '      test: [\'CMD-SHELL\', \'curl -fsS -H "Authorization: Bearer $$API_KEY" http://127.0.0.1:30010/healthz | grep -q "\\"ok\\":true"\']',
        "      interval: 10s",
        "      timeout: 5s",
        "      retries: 30",
        "      start_period: 30s",
    ]


def build_config(
    groups: list[list[dict[str, object]]],
    host: str,
    instance_id: str,
    base_port: int,
    cpu_per_group: int,
    memory_per_group_mb: int,
) -> list[dict[str, Any]]:
    instances = []
    for group_index, group in enumerate(groups):
        instances.append(
            {
                "id": f"{instance_id}-4h200-{group_index}",
                "host": host,
                "port": base_port + group_index,
                "internal_url": f"http://h3-api-{group_index}:30010",
                "group_index": group_index,
                "gpu_indexes": [int(gpu["index"]) for gpu in group],
                "gpu_uuids": [str(gpu["uuid"]) for gpu in group],
                "gpu_names": [str(gpu["name"]) for gpu in group],
                "gpu_memory_mb": [int(gpu["memory_mb"]) for gpu in group],
                "cpu": cpu_per_group,
                "memory_mb": memory_per_group_mb,
            }
        )
    return instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".generated")
    parser.add_argument("--data-root", default="/srv/minimax-h3-h200")
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--base-port", type=int, default=30010)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--sglang-image", default=os.getenv("SGLANG_IMAGE", ""))
    parser.add_argument("--api-image", default=os.getenv("API_IMAGE", ""))
    parser.add_argument("--allow-non-h200", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = detect_gpus()
    groups = partition_gpus(gpus, allow_non_h200=args.allow_non_h200)
    cpu_per_group, memory_per_group_mb = detect_host_resources(len(groups))

    compose = ["name: minimax-h3-4h200", "", "services:"]
    for group_index, group in enumerate(groups):
        compose.extend(sglang_service(group_index, group, args.data_root))
        compose.append("")
        compose.extend(
            api_service(
                group_index, group, args.data_root, args.advertise_host, args.base_port
            )
        )
        compose.append("")

    compose.extend(
        [
            "  h3-reporter:",
            "    image: ${REPORTER_IMAGE}",
            "    container_name: minimax-h3-h200-reporter",
            "    restart: unless-stopped",
            "    init: true",
            "    user: ${HOST_UID}:${HOST_GID}",
            "    env_file: ../.env",
            "    environment:",
            "      REPORTER_CONFIG: /config/instances.json",
            "      REPORTER_STATE: /state/status.json",
            "    volumes:",
            f"      - {quote(str(output_dir / 'instances.json') + ':/config/instances.json:ro')}",
            f"      - {args.data_root}/reporter:/state",
        ]
    )

    instances = build_config(
        groups,
        args.advertise_host,
        args.instance_id,
        args.base_port,
        cpu_per_group,
        memory_per_group_mb,
    )
    model_lock = json.loads((repo_root / "config/models.lock.json").read_text())
    reporter_config = {
        "node": {
            "instance_id": args.instance_id,
            "host": args.advertise_host,
            "gpu_count": len(gpus),
            "assigned_gpu_count": len(groups) * GPUS_PER_SERVICE,
            "service_count": len(groups),
        },
        "deployment": {
            "release_id": args.release_id,
            "sglang_image": args.sglang_image,
            "api_image": args.api_image,
            "model": model_lock,
        },
        "instances": instances,
    }
    (output_dir / "compose.yaml").write_text("\n".join(compose) + "\n")
    (output_dir / "instances.json").write_text(
        json.dumps(reporter_config, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "gpu-info.json").write_text(
        json.dumps(gpus, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"generated {len(groups)} services from {len(groups) * GPUS_PER_SERVICE}/{len(gpus)} GPUs "
        f"in {output_dir}"
    )


if __name__ == "__main__":
    main()
