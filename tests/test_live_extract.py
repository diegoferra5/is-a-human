import numpy as np

from is_a_human.analysis.features import extract_call_features_timed
from is_a_human.detect.tandem import load_tandem


def _short_stereo(duration_s: float = 1.5, sample_rate: int = 8000):
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    ch0 = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    ch1 = (0.2 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    return ch0, ch1, sample_rate


def test_timed_extract_records_layers_and_live_matches_full(tandem_model_path):
    model = load_tandem(tandem_model_path)
    ch0, ch1, sample_rate = _short_stereo()
    common = dict(
        anon_id="toy",
        label="human",
        split="val",
        ch0_caller=ch0,
        ch1_agent=ch1,
        sample_rate=sample_rate,
        transcript_turns=[{"channel": 1, "start": 0.0, "end": 0.4, "text": "su clave es 1234"}],
    )
    full_features, full_timings = extract_call_features_timed(**common, heavy=True)
    live_features, live_timings = extract_call_features_timed(**common, heavy=False)

    for timings in (full_timings, live_timings):
        assert timings["vad"] is not None
        assert timings["acoustic"] is not None
        assert timings["behavioral"] is not None
        assert timings["semantic"] is not None

    assert live_timings["acoustic_extras"] == 0.0
    assert full_timings["acoustic_extras"] >= 0.0

    full_p, full_views = model.predict(full_features)
    live_p, live_views = model.predict(live_features)
    assert live_p == full_p
    assert live_views == full_views
    assert {"acoustic", "behavioral"} <= set(live_views)
