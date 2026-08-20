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
