"""Conversational, acoustic, and recovery feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from time import perf_counter
from typing import Literal

import numpy as np

from is_a_human.analysis.acoustic import extract_acoustic_features
from is_a_human.analysis.interaction_physics import extract_interaction_physics_features
from is_a_human.analysis.micro_variation import extract_micro_variation_features
from is_a_human.analysis.recovery import extract_recovery_features
from is_a_human.analysis.semantic import extract_semantic_features
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

    caller_formant_f1_std: float
    caller_formant_f2_std: float
    caller_formant_f3_std: float
    caller_formant_volatility_mean: float
    caller_pitch_jitter: float
    caller_pitch_shimmer: float
    caller_f0_std: float
    caller_f0_cv: float
    caller_hnr_mean: float
    caller_hnr_std: float
    caller_hnr_cv: float
    caller_pause_entropy: float
    caller_lfcc_delta_delta_std: float

    caller_breath_event_rate: float
    caller_breath_gap_ratio: float
    cross_channel_energy_correlation: float
    caller_yield_decay_mean_db: float
    caller_yield_decay_std_db: float
    caller_agent_echo_correlation: float
    caller_agent_bleed_correlation: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def numeric_field_names(cls) -> tuple[str, ...]:
        return tuple(
            field.name
            for field in fields(cls)
            if field.name not in METADATA_FIELDS
        )


_MICRO_ZEROS = {
    "caller_formant_f1_std": 0.0,
    "caller_formant_f2_std": 0.0,
    "caller_formant_f3_std": 0.0,
    "caller_formant_volatility_mean": 0.0,
    "caller_pitch_jitter": 0.0,
    "caller_pitch_shimmer": 0.0,
    "caller_f0_std": 0.0,
    "caller_f0_cv": 0.0,
    "caller_hnr_mean": 0.0,
    "caller_hnr_std": 0.0,
    "caller_hnr_cv": 0.0,
    "caller_pause_entropy": 0.0,
    "caller_lfcc_delta_delta_std": 0.0,
}
_INTERACTION_ZEROS = {
    "caller_breath_event_rate": 0.0,
    "caller_breath_gap_ratio": 0.0,
    "cross_channel_energy_correlation": 0.0,
    "caller_yield_decay_mean_db": 0.0,
    "caller_yield_decay_std_db": 0.0,
    "caller_agent_echo_correlation": 0.0,
    "caller_agent_bleed_correlation": 0.0,
}

_ACOUSTIC_MARKERS = (
    "rms", "zcr", "spectral", "hf_lf", "crest", "intra_silence", "segment_length",
    "formant", "pitch", "f0", "hnr", "pause_entropy", "lfcc", "breath",
    "cross_channel", "yield_decay", "echo", "bleed",
)


def acoustic_feature_names() -> tuple[str, ...]:
    return tuple(
        name
        for name in CallFeatures.numeric_field_names()
        if name != "duration_s" and any(marker in name for marker in _ACOUSTIC_MARKERS)
    )


def behavioral_feature_names() -> tuple[str, ...]:
    acoustic = set(acoustic_feature_names())
    return tuple(
        name
        for name in CallFeatures.numeric_field_names()
        if name != "duration_s" and name not in acoustic
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


def _as_turn_segments(segments) -> tuple[TurnSegment, ...]:
    return tuple(
        TurnSegment(channel=segment.channel, start=segment.start, end=segment.end)
        for segment in segments
    )


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


def extract_call_features_timed(
    anon_id: str,
    label: str,
    split: str,
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    organizer_turns: tuple[TurnSegment, ...] | None = None,
    *,
    pipeline_result: PipelineResult | None = None,
    turns: tuple[TurnSegment, ...] | None = None,
    heavy: bool = True,
    transcript_turns: list[dict] | None = None,
) -> tuple[CallFeatures, dict[str, float | None]]:
    """Extract features and record VAD / acoustic / semantic / behavioural ms."""
    timings: dict[str, float | None] = {
        "vad": None,
        "acoustic": None,
        "semantic": None,
        "behavioral": None,
        "acoustic_extras": None,
    }

    if pipeline_result is None:
        started = perf_counter()
        result = process_call(ch0_caller, ch1_agent, sample_rate)
        timings["vad"] = _elapsed_ms(started)
    else:
        result = pipeline_result

    ledger = result.ledger
    metrics = result.metrics
    speech = turns if turns is not None else organizer_turns
    if speech is None:
        speech = _as_turn_segments(ledger.speech_segments)

    duration_s = max(len(ch0_caller), len(ch1_agent)) / sample_rate
    safe_duration = duration_s if duration_s > 0 else 1.0

    started = perf_counter()
    acoustic = extract_acoustic_features(ch0_caller, sample_rate, speech, duration_s)
    timings["acoustic"] = _elapsed_ms(started)

    started = perf_counter()
    caller_durations = _segment_durations(ledger, channel=0)
    agent_durations = _segment_durations(ledger, channel=1)
    caller_latencies = _response_latencies(ledger, responder_channel=0)
    agent_latencies = _response_latencies(ledger, responder_channel=1)
    caller_lat_mean, caller_lat_std, caller_lat_cv = _latency_stats(caller_latencies)
    agent_lat_mean, agent_lat_std, agent_lat_cv = _latency_stats(agent_latencies)
    overlap_events = sum(1 for event in ledger.events if event.type == TurnType.OVERLAP)
    recovery = extract_recovery_features(speech)
    timings["behavioral"] = _elapsed_ms(started)

    if transcript_turns:
        started = perf_counter()
        extract_semantic_features(transcript_turns)
        timings["semantic"] = _elapsed_ms(started)

    if heavy:
        started = perf_counter()
        micro = extract_micro_variation_features(ch0_caller, sample_rate, speech, duration_s)
        interaction = extract_interaction_physics_features(
            ch0_caller, ch1_agent, sample_rate, speech, duration_s
        )
        timings["acoustic_extras"] = _elapsed_ms(started)
    else:
        micro = dict(_MICRO_ZEROS)
        interaction = dict(_INTERACTION_ZEROS)
        timings["acoustic_extras"] = 0.0

    features = CallFeatures(
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
        **micro,
        **interaction,
    )
    return features, timings


def extract_call_features(
    anon_id: str,
    label: str,
    split: str,
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    organizer_turns: tuple[TurnSegment, ...] | None = None,
    *,
    pipeline_result: PipelineResult | None = None,
    turns: tuple[TurnSegment, ...] | None = None,
    heavy: bool = True,
    transcript_turns: list[dict] | None = None,
) -> CallFeatures:
    """Extract features. Default turns are VAD ledger segments, not organizer JSON."""
    features, _timings = extract_call_features_timed(
        anon_id,
        label,
        split,
        ch0_caller,
        ch1_agent,
        sample_rate,
        organizer_turns,
        pipeline_result=pipeline_result,
        turns=turns,
        heavy=heavy,
        transcript_turns=transcript_turns,
    )
    return features
