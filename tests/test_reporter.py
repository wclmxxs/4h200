import importlib.util
import json
from pathlib import Path

import httpx

MODULE_PATH = Path(__file__).resolve().parents[1] / "reporter/main.py"
SPEC = importlib.util.spec_from_file_location("h3_h200_reporter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def instance():
    return {
        "id": "i-test-4h200-1",
        "host": "16.78.214.130",
        "port": 30011,
        "internal_url": "http://h3-api-1:30010",
        "group_index": 1,
        "gpu_indexes": [4, 5, 6, 7],
        "gpu_uuids": ["GPU-4", "GPU-5", "GPU-6", "GPU-7"],
        "cpu": 48,
        "memory_mb": 512000,
    }


def test_probe_registers_one_four_gpu_instance():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://h3-api-1:30010/healthz"
        return httpx.Response(200, json={"ok": True, "healthy_workers": 1})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        bernard, detail = MODULE.probe_instance(client, instance())

    assert bernard["host"] == "16.78.214.130"
    assert bernard["ports"] == [30011]
    assert bernard["healthCheckResults"] == [{"alive": True}]
    assert bernard["containerInfos"]["h3-4h200-1"]["request"]["nvidia.com/gpu"] == 4
    assert detail["gpu_indexes"] == [4, 5, 6, 7]


def test_catalog_uses_h200_service_id(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"success": True})

    monkeypatch.setattr(MODULE, "CATALOG_URL", "https://gateway.test/report_catalog")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        MODULE.report_catalog(client, [{"id": "slot-0"}])

    assert captured["body"]["service_id"] == "Minimax-H3-AWS-H200"
    assert json.loads(captured["body"]["instances_json"]) == [{"id": "slot-0"}]
