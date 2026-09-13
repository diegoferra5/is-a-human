"""Behavioural features from turn timing (no audio, no words)."""

import pytest

from is_a_human.analysis.behavioral import extract_timing_features
from is_a_human.analysis.recovery import extract_recovery_features
from is_a_human.dataset.loader import TurnSegment
from is_a_human.turns.ledger import TurnEvent, TurnLedger, TurnType
from is_a_human.turns.metrics import compute_conversation_metrics

pytestmark = pytest.mark.behavioral


def _ledger(*events: TurnEvent) -> TurnLedger:
    return TurnLedger(events=events, speech_segments=(), frame_duration_s=0.032)


def test_talk_and_silence_ratios(report_metrics):
    ledger = _ledger(
        TurnEvent(channel=0, start=0.0, end=2.0, type=TurnType.SPEECH),
        TurnEvent(channel=None, start=2.0, end=2.5, type=TurnType.SILENCE),
        TurnEvent(channel=1, start=2.5, end=5.0, type=TurnType.SPEECH),
    )
    metrics = compute_conversation_metrics(ledger)
    report_metrics(
        caller_talk_time_s=metrics.caller_talk_time_s,
        agent_talk_time_s=metrics.agent_talk_time_s,
        silence_time_s=metrics.silence_time_s,
        overlap_time_s=metrics.overlap_time_s,
    )

    assert metrics.caller_talk_time_s == 2.0
    assert metrics.agent_talk_time_s == 2.5
    assert metrics.silence_time_s == 0.5
    assert metrics.overlap_time_s == 0.0


def test_overlap_counts_as_interruption(report_metrics):
    ledger = _ledger(
        TurnEvent(channel=1, start=0.0, end=1.0, type=TurnType.SPEECH),
        TurnEvent(channel=None, start=1.0, end=1.4, type=TurnType.OVERLAP),
        TurnEvent(channel=0, start=1.4, end=2.0, type=TurnType.SPEECH),
    )
    metrics = compute_conversation_metrics(ledger)
    report_metrics(
        overlap_time_s=metrics.overlap_time_s,
        interruption_count=metrics.interruption_count,
    )

    assert metrics.overlap_time_s == pytest.approx(0.4)
    assert metrics.interruption_count >= 1


def test_caller_response_latency(report_metrics):
    ledger = _ledger(
        TurnEvent(channel=1, start=0.0, end=1.0, type=TurnType.SPEECH),
        TurnEvent(channel=None, start=1.0, end=1.8, type=TurnType.SILENCE),
        TurnEvent(channel=0, start=1.8, end=2.5, type=TurnType.SPEECH),
    )
    metrics = compute_conversation_metrics(ledger)
    report_metrics(
        mean_agent_response_latency_s=metrics.mean_agent_response_latency_s,
        mean_caller_response_latency_s=metrics.mean_caller_response_latency_s,
    )

    assert metrics.mean_agent_response_latency_s == pytest.approx(0.8)


def test_floor_switch_positive_latency_excludes_barge_in(report_metrics):
    turns = (
        TurnSegment(channel=1, start=0.0, end=1.0),
        TurnSegment(channel=0, start=1.5, end=2.5),
        TurnSegment(channel=1, start=3.0, end=4.0),
        TurnSegment(channel=0, start=3.8, end=5.0),
    )
    feats = extract_timing_features(turns, duration_s=5.0)
    report_metrics(
        caller_response_latency_pos_mean_s=feats["caller_response_latency_pos_mean_s"],
        caller_response_latency_min_s=feats["caller_response_latency_min_s"],
        overlap_total_s=feats["overlap_total_s"],
    )

    assert feats["caller_response_latency_pos_mean_s"] == pytest.approx(0.5)
    assert feats["caller_response_latency_pos_median_s"] == pytest.approx(0.5)
    assert feats["caller_response_latency_min_s"] == pytest.approx(-0.2)
    assert feats["overlap_total_s"] == pytest.approx(0.2)
    assert feats["caller_segment_count"] == 2


def test_recovery_latency_after_agent_stop(report_metrics):
    turns = (
        TurnSegment(channel=1, start=0.0, end=1.0),
        TurnSegment(channel=0, start=1.5, end=2.5),
        TurnSegment(channel=1, start=3.0, end=4.0),
        TurnSegment(channel=0, start=4.2, end=5.0),
    )
    features = extract_recovery_features(turns)
    report_metrics(
        agent_aligned_recovery_count=features["agent_aligned_recovery_count"],
        agent_aligned_recovery_mean_s=features["agent_aligned_recovery_mean_s"],
    )

    assert features["agent_aligned_recovery_count"] == 2.0
    assert features["agent_aligned_recovery_mean_s"] == pytest.approx(0.35)


def test_timing_keys_are_call_feature_fields():
    from is_a_human.analysis.features import CallFeatures, behavioral_feature_names

    feats = extract_timing_features((), duration_s=1.0)
    numeric = set(CallFeatures.numeric_field_names())
    assert set(feats) <= numeric
    assert set(feats) <= set(behavioral_feature_names()) | {"duration_s"}
    assert feats["caller_response_latency_pos_mean_s"] == 0.0
