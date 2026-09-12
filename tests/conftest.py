import base64
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

CHALLENGE_DATASET_ROOT = Path("resources/challenge-dataset")


@pytest.fixture
def stereo_wav_bytes() -> bytes:
    sample_rate = 8000
    duration_s = 1.0
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    ch0 = 0.3 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    ch1 = 0.2 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    stereo = np.column_stack([ch0, ch1])

    buffer = io.BytesIO()
    sf.write(buffer, stereo, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@pytest.fixture
def stereo_wav_b64(stereo_wav_bytes: bytes) -> str:
    return base64.b64encode(stereo_wav_bytes).decode("ascii")


@pytest.fixture
def dataset_paths():
    if not (CHALLENGE_DATASET_ROOT / "manifest.csv").exists():
        pytest.skip("challenge-dataset resources not available")

    from is_a_human.dataset.paths import resolve_dataset_paths

    return resolve_dataset_paths(CHALLENGE_DATASET_ROOT)


@pytest.fixture
def real_call_sample(dataset_paths):
    from is_a_human.dataset.loader import load_call

    return load_call("call_0181ce113ebe", root=dataset_paths.root, load_audio=True)


@pytest.fixture
def real_call_b64(real_call_sample, dataset_paths):
    audio_path = dataset_paths.audio_dir / f"{real_call_sample.anon_id}.wav"
    return base64.b64encode(audio_path.read_bytes()).decode("ascii")
