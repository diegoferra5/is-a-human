from fastapi.testclient import TestClient

from is_a_human.api.app import create_app


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_endpoint_accepts_stereo_payload(stereo_wav_b64: str):
    client = TestClient(create_app())
    response = client.post("/detect", json={"audio_b64": stereo_wav_b64})
    assert response.status_code == 200
    payload = response.json()
    assert "is_synthetic" in payload
    assert "confidence" in payload
    assert 0.0 <= payload["confidence"] <= 1.0


def test_detect_endpoint_rejects_invalid_payload():
    client = TestClient(create_app())
    response = client.post("/detect", json={"audio_b64": "!!!"})
    assert response.status_code == 400
