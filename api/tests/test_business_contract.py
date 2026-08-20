import pytest
from app import business
from pydantic import ValidationError


def request(**updates):
    payload = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "A cinematic sunrise."}],
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
        "num_inference_steps": 6,
        "seed": 42,
    }
    payload.update(updates)
    return business.GenerationRequest.model_validate(payload)


def test_business_nfe_is_translated_to_sglang_sigma_grid():
    payload = business.to_upstream_request(request(num_inference_steps=6))
    assert payload["task"] == "t2va"
    assert payload["num_inference_steps"] == 7
    assert payload["target"] == {
        "short_edge": 768,
        "aspect_ratio": "16:9",
        "duration_seconds": 5.0,
    }


def test_fl2va_preserves_first_and_last_frame_order():
    value = request(
        content=[
            {"type": "text", "text": "Camera pushes in."},
            {
                "type": "image_url",
                "role": "last_frame",
                "image_url": {"url": "https://example.com/last.jpg"},
            },
            {
                "type": "image_url",
                "role": "first_frame",
                "image_url": {"url": "https://example.com/first.jpg"},
            },
        ],
        ratio="adaptive",
    )
    conditions = business.to_upstream_request(value)["conditions"]
    assert [item["frame_index"] for item in conditions] == [0, -1]


def test_reference_media_is_rejected_until_ref2va_is_deployed():
    with pytest.raises(ValidationError, match="ref2va is not deployed"):
        request(
            content=[
                {"type": "text", "text": "Animate this subject."},
                {
                    "type": "image_url",
                    "role": "reference_image",
                    "image_url": {"url": "https://example.com/ref.jpg"},
                },
            ]
        )


def test_turbo_profile_rejects_unvalidated_nfe():
    with pytest.raises(ValidationError, match="one of"):
        request(num_inference_steps=20)


def test_remote_media_host_policy_blocks_private_addresses():
    assert business.is_safe_remote_hostname("127.0.0.1") is False
    assert business.is_safe_remote_hostname("169.254.169.254") is False
    assert business.is_safe_remote_hostname("internal.byted.org") is True
    assert business.is_safe_remote_hostname("8.8.8.8") is True
