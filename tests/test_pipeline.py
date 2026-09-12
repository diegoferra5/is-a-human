import base64

import pytest

from is_a_human.audio.demux import demux_base64_telephony
from is_a_human.pipeline import process_call
from is_a_human.turns.ledger import TurnType
from is_a_human.turns.validation import compare_turn_segments


@pytest.mark.integration
def test_process_call_on_real_audio(real_call_sample):
    result = process_call(
        real_call_sample.ch0_caller,
        real_call_sample.ch1_agent,
        real_call_sample.sample_rate,
    )

    assert result.latency_ms > 0
    assert len(result.ledger.speech_segments) > 0
    assert len(result.ledger.events) > 0
    assert result.metrics.caller_talk_time_s > 0
    assert result.metrics.agent_talk_time_s > 0
    assert result.metrics.silence_time_s >= 0
    assert any(event.type == TurnType.SPEECH for event in result.ledger.events)


@pytest.mark.integration
def test_vad_iou_against_organizer_turns(real_call_sample):
    result = process_call(
        real_call_sample.ch0_caller,
        real_call_sample.ch1_agent,
        real_call_sample.sample_rate,
    )
    predicted = result.ledger.speech_segments

    caller_iou = compare_turn_segments(predicted, real_call_sample.turns, channel=0).iou
    agent_iou = compare_turn_segments(predicted, real_call_sample.turns, channel=1).iou

    assert caller_iou > 0.5
    assert agent_iou > 0.5


@pytest.mark.integration
def test_base64_round_trip_matches_direct_load(real_call_b64, real_call_sample):
    ch0, ch1, sample_rate = demux_base64_telephony(real_call_b64)

    assert sample_rate == real_call_sample.sample_rate
    assert ch0.shape == real_call_sample.ch0_caller.shape
    assert ch1.shape == real_call_sample.ch1_agent.shape
