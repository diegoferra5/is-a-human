from fastapi.testclient import TestClient

from is_a_human.api.app import create_app
from is_a_human.api.schemas import verdict_from_probability


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_verdict_confidence_is_certainty_in_label():
    is_synthetic, confidence = verdict_from_probability(0.87)
    assert is_synthetic is True
    assert confidence == 0.87
    is_synthetic, confidence = verdict_from_probability(0.13)
    assert is_synthetic is False
    assert confidence == 0.87
    recovered = confidence if is_synthetic else 1.0 - confidence
    assert recovered == 0.13


def test_detect_endpoint_accepts_judge_payload(stereo_wav_b64: str, tandem_model_path):
    client = TestClient(create_app(tandem_model_path))
    response = client.post(
        "/detect",
        json={
            "call_id": "call_0181ce113ebe",
            "audio_base64": stereo_wav_b64,
            "sample_rate": 8000,
            "channels": 2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["is_synthetic"], bool)
    assert 0.5 <= payload["confidence"] <= 1.0
    assert payload["views"] is not None
    assert "acoustic" in payload["views"]
    assert "behavioral" in payload["views"]


def test_detect_endpoint_accepts_legacy_audio_b64(stereo_wav_b64: str, tandem_model_path):
    client = TestClient(create_app(tandem_model_path))
    response = client.post("/detect", json={"audio_b64": stereo_wav_b64})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["is_synthetic"], bool)
    assert 0.5 <= payload["confidence"] <= 1.0


def test_detect_endpoint_rejects_invalid_payload(tandem_model_path):
    client = TestClient(create_app(tandem_model_path))
    response = client.post("/detect", json={"audio_base64": "!!!"})
    assert response.status_code == 400
