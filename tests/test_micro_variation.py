import numpy as np
import pytest

from is_a_human.analysis.interaction_physics import extract_interaction_physics_features
from is_a_human.analysis.micro_variation import extract_micro_variation_features
from is_a_human.dataset.loader import TurnSegment

pytestmark = pytest.mark.acoustic


def _sine_burst(start_s: float, end_s: float, sample_rate: int, freq: float = 180.0) -> np.ndarray:
    duration = end_s - start_s
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_micro_variation_features_on_synthetic_call():
    sample_rate = 8000
    duration_s = 2.0
    n = int(duration_s * sample_rate)
    ch0 = np.zeros(n, dtype=np.float32)
    ch1 = np.zeros(n, dtype=np.float32)
    ch0[: int(0.8 * sample_rate)] = _sine_burst(0, 0.8, sample_rate)
    ch0[int(1.2 * sample_rate) :] = _sine_burst(1.2, 2.0, sample_rate)
    ch1[int(0.9 * sample_rate) : int(1.1 * sample_rate)] = _sine_burst(0, 0.2, sample_rate, freq=120.0)

    turns = (
        TurnSegment(channel=0, start=0.0, end=0.8),
        TurnSegment(channel=1, start=0.9, end=1.1),
        TurnSegment(channel=0, start=1.2, end=2.0),
    )

    micro = extract_micro_variation_features(ch0, sample_rate, turns, duration_s)
    interaction = extract_interaction_physics_features(ch0, ch1, sample_rate, turns, duration_s)

    assert micro["caller_pause_entropy"] >= 0.0
    assert micro["caller_formant_volatility_mean"] >= 0.0
    assert micro["caller_hnr_mean"] >= 0.0
    assert interaction["cross_channel_energy_correlation"] >= -1.0
    assert interaction["caller_agent_echo_correlation"] >= 0.0
    assert interaction["caller_agent_bleed_correlation"] >= 0.0
    assert 0.0 <= interaction["caller_breath_gap_ratio"] <= 1.0
