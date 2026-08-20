import pytest
from app import main
from fastapi import HTTPException


def test_job_path_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path)
    with pytest.raises(HTTPException) as raised:
        main.job_file("../escape")
    assert raised.value.status_code == 400


def test_metadata_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path)
    main.save_metadata("video_123", {"id": "video_123", "business": {"nfe": 6}})
    assert main.load_metadata("video_123") == {
        "id": "video_123",
        "business": {"nfe": 6},
    }


def test_job_status_is_only_written_on_transition(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "JOB_ROOT", tmp_path)
    metadata = {"id": "video_123", "created_at": 100}
    main.save_metadata("video_123", metadata)

    observed = main.record_job_status(
        "video_123", "queued", metadata=metadata, now=100
    )
    assert observed["_watchdog"] == {
        "status": "queued",
        "status_changed_at": 100,
        "terminal": False,
    }
    unchanged = main.record_job_status(
        "video_123", "queued", metadata=observed, now=200
    )
    assert unchanged["_watchdog"]["status_changed_at"] == 100
    completed = main.record_job_status(
        "video_123", "completed", metadata=unchanged, now=300
    )
    assert completed["_watchdog"]["terminal"] is True
    assert completed["_watchdog"]["status_changed_at"] == 300
