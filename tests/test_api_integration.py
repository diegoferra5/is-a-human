import pytest
from fastapi.testclient import TestClient

from is_a_human.api.app import create_app


@pytest.mark.integration
def test_detect_with_real_call_audio(real_call_b64):
    client = TestClient(create_app())
    response = client.post("/detect", json={"audio_b64": real_call_b64})

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["is_synthetic"], bool)
    assert 0.0 <= payload["confidence"] <= 1.0
