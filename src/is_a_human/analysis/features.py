"""Conversational, acoustic, and recovery feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from is_a_human.analysis.acoustic import extract_acoustic_features
from is_a_human.analysis.recovery import extract_recovery_features
from is_a_human.dataset.loader import TurnSegment
from is_a_human.pipeline import PipelineResult, process_call
from is_a_human.turns.ledger import TurnLedger, TurnType

METADATA_FIELDS = {"anon_id", "label", "split"}


@dataclass(frozen=True)
class CallFeatures:
    anon_id: str
    label: str
    split: str
    duration_s: float

    caller_talk_ratio: float
    agent_talk_ratio: float
    overlap_ratio: float
    silence_ratio: float

    caller_segment_count: int
    agent_segment_count: int
    overlap_event_count: int

    caller_utterance_mean_s: float
    caller_utterance_std_s: float
    agent_utterance_mean_s: float
    agent_utterance_std_s: float

    caller_response_latency_mean_s: float
    caller_response_latency_std_s: float
    caller_response_latency_cv: float

    agent_response_latency_mean_s: float
    agent_response_latency_std_s: float
    agent_response_latency_cv: float

    caller_rms_mean: float
    caller_rms_std: float
    caller_rms_cv: float
    caller_zcr_mean: float
    caller_zcr_std: float
    caller_spectral_centroid_mean: float
    caller_spectral_centroid_std: float
    caller_spectral_flatness_mean: float
    caller_spectral_flatness_std: float
    caller_hf_lf_ratio_mean: float
    caller_hf_lf_ratio_std: float
    caller_crest_factor_mean: float
    caller_crest_factor_std: float
    caller_crest_factor_cv: float
    caller_intra_silence_gap_mean_s: float
    caller_intra_silence_gap_cv: float
    caller_segment_length_cv: float

    agent_aligned_recovery_count: float
    agent_aligned_recovery_mean_s: float
    agent_aligned_recovery_std_s: float
    agent_aligned_recovery_cv: float
    agent_aligned_recovery_utterance_mean_s: float
    agent_aligned_recovery_utterance_std_s: float
    agent_aligned_recovery_utterance_cv: float
    post_overlap_recovery_mean_s: float
    post_overlap_recovery_cv: float
    post_overlap_recovery_count: float
    caller_barge_in_count: float
    caller_backchannel_count: float
    agent_turn_fragmentation: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def numeric_field_names(cls) -> tuple[str, ...]:
        return tuple(
            field.name
            for field in fields(cls)
            if field.name not in METADATA_FIELDS
        )


def numeric_feature_vector(features: CallFeatures) -> np.ndarray:
    return np.array([float(getattr(features, name)) for name in CallFeatures.numeric_field_names()])


def _segment_durations(ledger: TurnLedger, channel: int) -> list[float]:
    return [
        segment.end - segment.start
        for segment in ledger.speech_segments
        if segment.channel == channel
    ]


def _response_latencies(ledger: TurnLedger, responder_channel: int) -> list[float]:
    other_channel = 1 - responder_channel
    latencies: list[float] = []

    other_stops = [
        event.end
        for event in ledger.events
        if event.type == TurnType.SPEECH and event.channel == other_channel
    ]
    responder_starts = [
        event.start
        for event in ledger.events
        if event.type == TurnType.SPEECH and event.channel == responder_channel
    ]

    for stop_time in other_stops:
        next_starts = [start for start in responder_starts if start > stop_time]
        if next_starts:
            latencies.append(next_starts[0] - stop_time)

    return latencies


def _latency_stats(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(latencies))
    std = float(np.std(latencies))
    cv = std / mean if mean > 0 else 0.0
    return mean, std, cv


def extract_call_features(
    anon_id: str,
    label: str,
    split: str,
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    organizer_turns: tuple[TurnSegment, ...],
    *,
    pipeline_result: PipelineResult | None = None,
) -> CallFeatures:
    """Extract full feature set from a call."""
    result = pipeline_result or process_call(ch0_caller, ch1_agent, sample_rate)
    ledger = result.ledger
    metrics = result.metrics

    duration_s = max(len(ch0_caller), len(ch1_agent)) / sample_rate
    safe_duration = duration_s if duration_s > 0 else 1.0

    caller_durations = _segment_durations(ledger, channel=0)
    agent_durations = _segment_durations(ledger, channel=1)
    caller_latencies = _response_latencies(ledger, responder_channel=0)
    agent_latencies = _response_latencies(ledger, responder_channel=1)

    caller_lat_mean, caller_lat_std, caller_lat_cv = _latency_stats(caller_latencies)
    agent_lat_mean, agent_lat_std, agent_lat_cv = _latency_stats(agent_latencies)

    overlap_events = sum(1 for event in ledger.events if event.type == TurnType.OVERLAP)

    acoustic = extract_acoustic_features(ch0_caller, sample_rate, organizer_turns, duration_s)
    recovery = extract_recovery_features(organizer_turns)

    return CallFeatures(
        anon_id=anon_id,
        label=label,
        split=split,
        duration_s=duration_s,
        caller_talk_ratio=metrics.caller_talk_time_s / safe_duration,
        agent_talk_ratio=metrics.agent_talk_time_s / safe_duration,
        overlap_ratio=metrics.overlap_time_s / safe_duration,
        silence_ratio=metrics.silence_time_s / safe_duration,
        caller_segment_count=len(caller_durations),
        agent_segment_count=len(agent_durations),
        overlap_event_count=overlap_events,
        caller_utterance_mean_s=float(np.mean(caller_durations)) if caller_durations else 0.0,
        caller_utterance_std_s=float(np.std(caller_durations)) if caller_durations else 0.0,
        agent_utterance_mean_s=float(np.mean(agent_durations)) if agent_durations else 0.0,
        agent_utterance_std_s=float(np.std(agent_durations)) if agent_durations else 0.0,
        caller_response_latency_mean_s=caller_lat_mean,
        caller_response_latency_std_s=caller_lat_std,
        caller_response_latency_cv=caller_lat_cv,
        agent_response_latency_mean_s=agent_lat_mean,
        agent_response_latency_std_s=agent_lat_std,
        agent_response_latency_cv=agent_lat_cv,
        **acoustic,
        **recovery,
    )
