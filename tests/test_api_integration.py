import pytest
from fastapi.testclient import TestClient

from is_a_human.api.app import create_app


@pytest.mark.integration
def test_detect_with_real_call_audio(real_call_b64, tandem_model_path):
    client = TestClient(create_app(tandem_model_path))
    response = client.post(
        "/detect",
        json={
            "call_id": "call_0181ce113ebe",
            "audio_base64": real_call_b64,
            "sample_rate": 8000,
            "channels": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["is_synthetic"], bool)
    assert 0.5 <= payload["confidence"] <= 1.0
    assert payload["views"]
