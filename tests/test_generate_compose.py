import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/generate_compose.py"
SPEC = importlib.util.spec_from_file_location("generate_compose", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def h200s(count: int):
    return [
        {
            "index": index,
            "uuid": f"GPU-{index}",
            "name": "NVIDIA H200",
            "memory_mb": 143771,
        }
        for index in range(count)
    ]


def test_eight_h200s_form_two_disjoint_groups():
    groups = MODULE.partition_gpus(h200s(8))
    assert [[gpu["index"] for gpu in group] for group in groups] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_incomplete_group_is_ignored(capsys):
    groups = MODULE.partition_gpus(h200s(10))
    assert len(groups) == 2
    assert "ignoring GPUs [8, 9]" in capsys.readouterr().err


def test_non_h200_fails_closed():
    gpus = h200s(4)
    gpus[2]["name"] = "RTX PRO 6000"
    with pytest.raises(SystemExit, match="must be H200"):
        MODULE.partition_gpus(gpus)


def test_main_renders_two_registered_services(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "detect_gpus", lambda: h200s(8))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_compose.py",
            "--output-dir",
            str(tmp_path),
            "--data-root",
            "/srv/h3-data",
            "--model-cache-root",
            "/mnt/model-ebs/hf-cache",
            "--advertise-host",
            "16.78.214.130",
            "--instance-id",
            "i-test",
            "--release-id",
            "release-test",
            "--sglang-image",
            "sglang:test",
            "--api-image",
            "api:test",
        ],
    )
    MODULE.main()
    compose = yaml.safe_load((tmp_path / "compose.yaml").read_text())
    config = json.loads((tmp_path / "instances.json").read_text())

    assert len(config["instances"]) == 2
    assert [item["port"] for item in config["instances"]] == [30010, 30011]
    assert [item["host"] for item in config["instances"]] == [
        "16.78.214.130",
        "16.78.214.130",
    ]
    assert config["instances"][1]["gpu_indexes"] == [4, 5, 6, 7]
    assert config["instances"][1]["id"] == "i-test-4h200-1"
    assert len(compose["services"]) == 7
    reservations = compose["services"]["h3-sglang-1"]["deploy"]["resources"][
        "reservations"
    ]["devices"]
    assert reservations[0]["device_ids"] == ["4", "5", "6", "7"]
    assert compose["services"]["h3-api-1"]["ports"] == ["0.0.0.0:30011:30010"]
    assert compose["services"]["h3-sglang-0"]["image"] == "${SGLANG_IMAGE}"
    assert compose["services"]["h3-sglang-1"]["image"] == "${SGLANG_IMAGE}"
    assert [item["attention_profile"] for item in config["instances"]] == [
        "sage_attn",
        "sage_attn",
    ]
    assert (
        "/mnt/model-ebs/hf-cache:/cache/huggingface"
        in compose["services"]["h3-sglang-0"]["volumes"]
    )
    assert compose["services"]["h3-cleaner"]["environment"] == {
        "CLEANUP_ROOT": "/slots",
        "CLEANUP_STATE": "/state/status.json",
    }
    assert (
        compose["services"]["h3-sglang-0"]["environment"][
            "PYTORCH_CUDA_ALLOC_CONF"
        ]
        == "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    )
    watchdog = compose["services"]["h3-watchdog"]
    assert watchdog["image"] == "${WATCHDOG_IMAGE}"
    assert "/var/run/docker.sock:/var/run/docker.sock" in watchdog["volumes"]
    assert "/srv/h3-data/slots:/slots:ro" in watchdog["volumes"]


def test_sglang_command_contains_static_lora_and_four_gpu_topology():
    command = MODULE.sglang_command()
    assert "--model-variant fl2va" in command
    assert "--num-gpus 4" in command
    assert 'lora_path="$$LORA_REPO"' in command
    assert 'lora_path="$$LORA_LOCAL_PATH"' in command
    assert '--lora-path "$$lora_path"' in command
    assert '--lora-weight-name "$$LORA_WEIGHT"' in command
    assert 'attention_backend="$${ATTENTION_BACKEND:-fa}"' in command
    assert (
        'component_attention_backends="$${COMPONENT_ATTENTION_BACKENDS:-transformer=sage_attn}"'
        in command
    )
    assert '--attention-backend "$$attention_backend"' in command
    assert '--component-attention-backends "$$component_attention_backends"' in command
    assert 'attention_backend_config="$${ATTENTION_BACKEND_CONFIG:-}"' in command
    assert '--attention-backend-config "$$attention_backend_config"' in command
    assert 'warmup_steps="$${WARMUP_STEPS:-}"' in command
    assert '--warmup-steps "$$warmup_steps"' in command
    assert 'quantization="$${QUANTIZATION:-}"' in command
    assert '--quantization "$$quantization"' in command
    assert 'if [[ -n "$$component_attention_backends" ]]; then' in command
    assert 'exec "$${args[@]}"' in command


def test_optimization_stack_applies_to_every_four_gpu_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "detect_gpus", lambda: h200s(8))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_compose.py",
            "--output-dir",
            str(tmp_path),
            "--data-root",
            "/srv/h3-data",
            "--advertise-host",
            "16.78.214.130",
            "--instance-id",
            "i-test",
            "--release-id",
            "release-test",
            "--sglang-image",
            "sglang:sage",
            "--sglang-sol-image",
            "sglang:sol",
            "--api-image",
            "api:test",
            "--optimization-stack-enabled",
            "--sol-component-attention-backends",
            "text_encoder=torch_sdpa,audio_vae=fa,video_vae=fa,transformer=sol_attn",
            "--sol-attention-backend-config",
            "dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5",
        ],
    )

    MODULE.main()
    compose = yaml.safe_load((tmp_path / "compose.yaml").read_text())
    config = json.loads((tmp_path / "instances.json").read_text())

    assert compose["services"]["h3-sglang-0"]["deploy"]["resources"]["reservations"][
        "devices"
    ][0]["device_ids"] == ["0", "1", "2", "3"]
    assert compose["services"]["h3-sglang-1"]["deploy"]["resources"]["reservations"][
        "devices"
    ][0]["device_ids"] == ["4", "5", "6", "7"]
    for slot in (0, 1):
        worker = compose["services"][f"h3-sglang-{slot}"]
        assert worker["image"] == "${SGLANG_SOL_IMAGE}"
        env = worker["environment"]
        assert env["COMPONENT_ATTENTION_BACKENDS"].endswith("transformer=sol_attn}")
        assert "audio_vae=fa" in env["COMPONENT_ATTENTION_BACKENDS"]
        assert "video_vae=fa" in env["COMPONENT_ATTENTION_BACKENDS"]
        assert "dense_steps=0" in env["ATTENTION_BACKEND_CONFIG"]
        assert "tau=1.5" in env["ATTENTION_BACKEND_CONFIG"]
        assert env["ATTENTION_BACKEND"] == "sol_attn"
        assert env["SOL_ATTN_STRICT"] == "${SOL_ATTN_STRICT:-1}"
        assert env["WARMUP_STEPS"] == "${SOL_WARMUP_STEPS:-3}"
        assert env["QUANTIZATION"] == "${SOL_QUANTIZATION:-fp8}"
        assert env["LORA_MERGE_MODE"] == "${SOL_LORA_MERGE_MODE:-dynamic}"
        assert env["SGLANG_CACHE_DIT_ENABLED"] == "${SOL_CACHE_DIT_ENABLED:-true}"
        assert env["SGLANG_CACHE_DIT_WARMUP"] == "${SOL_CACHE_DIT_WARMUP:-1}"
        assert env["SGLANG_CACHE_DIT_RDT"] == "${SOL_CACHE_DIT_RDT:-0.12}"
        assert env["SGLANG_CACHE_DIT_MC"] == "${SOL_CACHE_DIT_MC:-3}"
        assert (
            compose["services"][f"h3-api-{slot}"]["environment"]["ATTENTION_BACKEND"]
            == "sol_attn"
        )
    assert [item["attention_profile"] for item in config["instances"]] == [
        "sol_attn",
        "sol_attn",
    ]
    assert [item["optimization_profile"] for item in config["instances"]] == [
        "sol_attn_fp8_cache_dit",
        "sol_attn_fp8_cache_dit",
    ]
    stack = config["deployment"]["optimization_stack"]
    assert stack["enabled"] is True
    assert stack["quantization"] == "fp8"
    assert stack["lora_merge_mode"] == "dynamic"
    assert stack["cache_dit"] == {
        "enabled": "true",
        "fn": "1",
        "bn": "0",
        "warmup": "1",
        "rdt": "0.12",
        "mc": "3",
    }


def test_optimization_stack_supports_one_four_gpu_group(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "detect_gpus", lambda: h200s(4))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_compose.py",
            "--output-dir",
            str(tmp_path),
            "--advertise-host",
            "16.78.214.130",
            "--instance-id",
            "i-test",
            "--release-id",
            "release-test",
            "--sglang-sol-image",
            "sglang:sol",
            "--optimization-stack-enabled",
        ],
    )

    MODULE.main()
    compose = yaml.safe_load((tmp_path / "compose.yaml").read_text())
    assert compose["services"]["h3-sglang-0"]["image"] == "${SGLANG_SOL_IMAGE}"
