from unittest.mock import patch

import numpy as np
import pytest

from is_a_human.turns.ledger import TurnType, build_turn_ledger
from is_a_human.turns.metrics import compute_conversation_metrics
from is_a_human.turns.validation import compare_turn_segments
from is_a_human.turns.vad import VadSegment
from is_a_human.dataset.loader import TurnSegment

pytestmark = pytest.mark.vad


@patch("is_a_human.turns.ledger.detect_dual_channel_segments")
def test_build_turn_ledger_on_alternating_speech(mock_detect):
    sample_rate = 8000
    num_samples = int(sample_rate * 4.0)
    ch0 = np.zeros(num_samples, dtype=np.float32)
    ch1 = np.zeros(num_samples, dtype=np.float32)

    mock_detect.return_value = [
        VadSegment(channel=0, start=0.0, end=1.0),
        VadSegment(channel=1, start=1.2, end=2.0),
        VadSegment(channel=0, start=2.2, end=3.0),
    ]

    ledger = build_turn_ledger(ch0, ch1, sample_rate)
    metrics = compute_conversation_metrics(ledger)

    assert len(ledger.events) > 0
    assert metrics.caller_talk_time_s > 0
    assert metrics.agent_talk_time_s > 0
    assert any(event.type == TurnType.SILENCE for event in ledger.events)


def test_compare_turn_segments_iou():
    predicted = [
        VadSegment(channel=0, start=0.0, end=2.0),
        VadSegment(channel=1, start=1.0, end=3.0),
    ]
    reference = [
        TurnSegment(channel=0, start=0.5, end=2.5),
        TurnSegment(channel=1, start=1.0, end=2.0),
    ]

    caller = compare_turn_segments(predicted, reference, channel=0)
    agent = compare_turn_segments(predicted, reference, channel=1)

    assert caller.iou > 0.5
    assert agent.iou == 0.5
