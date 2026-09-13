import base64
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest_plugins = ["html_reporter"]

_DATASET_CANDIDATES = (
    Path("resources/challenge-dataset"),
    Path("resources/hackmty26-main"),
)


def _resolve_test_dataset_root() -> Path | None:
    for root in _DATASET_CANDIDATES:
        if (root / "manifest.csv").exists() and (root / "turns").is_dir():
            return root
    return None


@pytest.fixture
def report_metrics(request):
    """Attach extra numbers to the HTML report row for this test."""

    def _record(**values):
        for key, value in values.items():
            request.node.user_properties.append((key, value))

    return _record


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
    root = _resolve_test_dataset_root()
    if root is None:
        pytest.skip("dataset resources not available")

    from is_a_human.dataset.paths import resolve_dataset_paths

    return resolve_dataset_paths(root)


@pytest.fixture
def real_call_sample(dataset_paths):
    from is_a_human.dataset.loader import load_call

    return load_call("call_0181ce113ebe", root=dataset_paths.root, load_audio=True)


@pytest.fixture
def tandem_model_path(tmp_path):
    from is_a_human.analysis.features import CallFeatures
    from is_a_human.detect.tandem import save_tandem, train_tandem_from_rows

    def _row(label: str, talk: float, agent: float, rms: float) -> CallFeatures:
        base = {name: 0.0 for name in CallFeatures.numeric_field_names()}
        zcr = 0.9 if label == "human" else 0.1
        base.update(
            caller_talk_ratio=talk,
            agent_talk_ratio=agent,
            caller_rms_mean=rms,
            caller_zcr_std=zcr,
            caller_rms_cv=zcr,
        )
        return CallFeatures(anon_id=f"{label}_{talk}_{rms}", label=label, split="train", **base)

    train = [
        _row("human", 0.15, 0.55, 0.05),
        _row("human", 0.16, 0.57, 0.06),
        _row("human", 0.14, 0.56, 0.04),
        _row("synthetic", 0.40, 0.30, 0.40),
        _row("synthetic", 0.42, 0.28, 0.38),
        _row("synthetic", 0.38, 0.32, 0.42),
    ]
    val = [
        _row("human", 0.17, 0.54, 0.07),
        _row("synthetic", 0.41, 0.29, 0.39),
    ]
    path = tmp_path / "tandem.json"
    save_tandem(train_tandem_from_rows(train, val), path)
    return path


@pytest.fixture
def real_call_b64(real_call_sample, dataset_paths):
    audio_path = dataset_paths.audio_dir / f"{real_call_sample.anon_id}.wav"
    return base64.b64encode(audio_path.read_bytes()).decode("ascii")
