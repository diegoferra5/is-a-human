import pytest

from is_a_human.analysis.recovery import extract_recovery_features
from is_a_human.dataset.loader import TurnSegment

pytestmark = pytest.mark.behavioral


def test_recovery_features_after_agent_stop():
    turns = (
        TurnSegment(channel=1, start=0.0, end=2.0),
        TurnSegment(channel=0, start=3.0, end=4.0),
        TurnSegment(channel=1, start=5.0, end=6.0),
        TurnSegment(channel=0, start=6.5, end=7.0),
    )
    features = extract_recovery_features(turns)

    assert features["agent_aligned_recovery_count"] == 2.0
    assert features["agent_aligned_recovery_mean_s"] == 0.75
    assert features["agent_aligned_recovery_cv"] >= 0.0


def test_barge_in_detection():
    turns = (
        TurnSegment(channel=1, start=0.0, end=5.0),
        TurnSegment(channel=0, start=2.0, end=2.8),
    )
    features = extract_recovery_features(turns)

    assert features["caller_barge_in_count"] == 1.0
    assert features["caller_backchannel_count"] == 1.0
