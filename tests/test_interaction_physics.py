import numpy as np
import pytest

from is_a_human.analysis.interaction_physics import (
    _agent_only_intervals,
    _gaps_before_long_utterances,
    extract_interaction_physics_features,
)
from is_a_human.dataset.loader import TurnSegment

pytestmark = pytest.mark.acoustic


def test_agent_only_intervals_exclude_caller_overlap():
    caller = [(0.0, 3.0), (5.0, 8.0)]
    agent = [(1.0, 6.0)]
    intervals = _agent_only_intervals(caller, agent, duration_s=10.0)
    assert intervals == [(3.0, 5.0)]


def test_gaps_before_long_utterances():
    caller = [(0.0, 0.5), (1.0, 4.0), (5.0, 5.5)]
    gaps = _gaps_before_long_utterances(caller, min_utterance_s=2.0)
    assert gaps == [(0.5, 1.0)]


def test_agent_bleed_detects_lagged_copy():
    sample_rate = 8000
    duration_s = 1.0
    n = int(duration_s * sample_rate)
    lag = 40
    agent = np.random.default_rng(0).normal(0, 0.1, n).astype(np.float32)
    caller = np.zeros(n, dtype=np.float32)
    caller[lag:] = agent[:-lag] * 0.2

    turns = (
        TurnSegment(channel=1, start=0.0, end=1.0),
    )
    features = extract_interaction_physics_features(
        caller, agent, sample_rate, turns, duration_s
    )
    assert features["caller_agent_bleed_correlation"] > 0.05
