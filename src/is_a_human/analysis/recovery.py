"""Agent-turn-aligned recovery features from organizer turn segments."""

from __future__ import annotations

import numpy as np

from is_a_human.dataset.loader import TurnSegment


def _segments_by_channel(turns: tuple[TurnSegment, ...], channel: int) -> list[tuple[float, float]]:
    return sorted(
        [(segment.start, segment.end) for segment in turns if segment.channel == channel],
        key=lambda item: item[0],
    )


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _safe_stats(values: list[float]) -> tuple[float, float, float, int]:
    if not values:
        return 0.0, 0.0, 0.0, 0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = std / mean if mean > 0 else 0.0
    return mean, std, cv, len(values)


def _recovery_latencies(
    trigger_ends: list[float],
    responder_segments: list[tuple[float, float]],
) -> list[float]:
    latencies: list[float] = []
    for end_time in trigger_ends:
        next_starts = [start for start, _ in responder_segments if start > end_time]
        if next_starts:
            latencies.append(next_starts[0] - end_time)
    return latencies


def _recovery_utterance_lengths(
    trigger_ends: list[float],
    responder_segments: list[tuple[float, float]],
) -> list[float]:
    lengths: list[float] = []
    for end_time in trigger_ends:
        for start, end in responder_segments:
            if start > end_time:
                lengths.append(end - start)
                break
    return lengths


def extract_recovery_features(organizer_turns: tuple[TurnSegment, ...]) -> dict[str, float]:
    """Measure caller recovery after agent stops, overlap, and agent interruptions."""
    caller_segments = _segments_by_channel(organizer_turns, channel=0)
    agent_segments = _segments_by_channel(organizer_turns, channel=1)

    agent_stop_recovery = _recovery_latencies(
        [end for _, end in agent_segments],
        caller_segments,
    )
    agent_stop_utterances = _recovery_utterance_lengths(
        [end for _, end in agent_segments],
        caller_segments,
    )

    overlap_triggers: list[float] = []
    barge_in_count = 0
    backchannel_count = 0

    for caller_start, caller_end in caller_segments:
        caller_duration = caller_end - caller_start
        overlapped_agent = False
        for agent_start, agent_end in agent_segments:
            if _overlaps((caller_start, caller_end), (agent_start, agent_end)):
                overlapped_agent = True
                overlap_triggers.append(agent_end)
                if caller_start >= agent_start:
                    barge_in_count += 1
        if overlapped_agent and caller_duration < 1.0:
            backchannel_count += 1

    overlap_recovery = _recovery_latencies(overlap_triggers, caller_segments)

    agent_stop_mean, agent_stop_std, agent_stop_cv, agent_stop_n = _safe_stats(agent_stop_recovery)
    utter_mean, utter_std, utter_cv, _ = _safe_stats(agent_stop_utterances)
    overlap_mean, overlap_std, overlap_cv, overlap_n = _safe_stats(overlap_recovery)

    agent_durations = [end - start for start, end in agent_segments]
    agent_frag = (
        len(agent_segments) / (sum(agent_durations) / len(agent_durations))
        if agent_durations
        else 0.0
    )

    return {
        "agent_aligned_recovery_count": float(agent_stop_n),
        "agent_aligned_recovery_mean_s": agent_stop_mean,
        "agent_aligned_recovery_std_s": agent_stop_std,
        "agent_aligned_recovery_cv": agent_stop_cv,
        "agent_aligned_recovery_utterance_mean_s": utter_mean,
        "agent_aligned_recovery_utterance_std_s": utter_std,
        "agent_aligned_recovery_utterance_cv": utter_cv,
        "post_overlap_recovery_mean_s": overlap_mean,
        "post_overlap_recovery_cv": overlap_cv,
        "post_overlap_recovery_count": float(overlap_n),
        "caller_barge_in_count": float(barge_in_count),
        "caller_backchannel_count": float(backchannel_count),
        "agent_turn_fragmentation": agent_frag,
    }
