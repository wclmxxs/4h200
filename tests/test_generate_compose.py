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
    assert len(compose["services"]) == 6
    reservations = compose["services"]["h3-sglang-1"]["deploy"]["resources"][
        "reservations"
    ]["devices"]
    assert reservations[0]["device_ids"] == ["4", "5", "6", "7"]
    assert compose["services"]["h3-api-1"]["ports"] == ["0.0.0.0:30011:30010"]
    assert compose["services"]["h3-cleaner"]["environment"] == {
        "CLEANUP_ROOT": "/slots",
        "CLEANUP_STATE": "/state/status.json",
    }


def test_sglang_command_contains_static_lora_and_four_gpu_topology():
    command = MODULE.sglang_command()
    assert "--model-variant fl2va" in command
    assert "--num-gpus 4" in command
    assert 'lora_path="$$LORA_REPO"' in command
    assert 'lora_path="$$LORA_LOCAL_PATH"' in command
    assert '--lora-path "$$lora_path"' in command
    assert '--lora-weight-name "$$LORA_WEIGHT"' in command
    assert 'attention_backend="$${ATTENTION_BACKEND:-sage_attn}"' in command
    assert '--attention-backend "$$attention_backend"' in command
    assert '--component-attention-backends "$$component_attention_backends"' in command
    assert '"$$attention_backend" == "sage_attn"' in command
    assert 'exec "$${args[@]}"' in command
