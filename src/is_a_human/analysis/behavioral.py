"""Turn-timing features: who talks when, not what they sound like.

Floor-switch latency is the Max-branch signal: walk the merged timeline and
record the gap when the floor changes speaker. Negative gaps are barge-ins.
Positive-only caller gaps (resp_lat_pos_*) are the strongest behavioural
separator — synthetics wait longer and more uniformly after the agent stops.
"""

from __future__ import annotations

from statistics import mean, median, pstdev
from typing import Sequence

from is_a_human.dataset.loader import TurnSegment

CALLER_CH = 0
AGENT_CH = 1


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0}
    m = mean(values)
    sd = pstdev(values) if len(values) > 1 else 0.0
    return {
        "mean": float(m),
        "std": float(sd),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "cv": float(sd / m) if m else 0.0,
    }


def _overlap(a: TurnSegment, b: TurnSegment) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def _floor_switch_gaps(ordered: Sequence[TurnSegment]) -> tuple[list[float], list[float]]:
    caller: list[float] = []
    agent: list[float] = []
    prev: TurnSegment | None = None
    for turn in ordered:
        if prev is not None and prev.channel != turn.channel:
            gap = turn.start - prev.end
            if turn.channel == CALLER_CH:
                caller.append(gap)
            else:
                agent.append(gap)
        prev = turn
    return caller, agent


def extract_timing_features(
    turns: Sequence[TurnSegment],
    duration_s: float,
) -> dict[str, float]:
    """Max-style timing vector from in-memory turn segments (no JSON, no audio)."""
    caller = sorted((t for t in turns if t.channel == CALLER_CH), key=lambda t: t.start)
    agent = sorted((t for t in turns if t.channel == AGENT_CH), key=lambda t: t.start)
    ordered = sorted(turns, key=lambda t: t.start)

    caller_dur = [t.end - t.start for t in caller]
    agent_dur = [t.end - t.start for t in agent]
    caller_speech = sum(caller_dur)
    agent_speech = sum(agent_dur)
    duration_s = max(duration_s, 1e-6)

    caller_lat, agent_lat = _floor_switch_gaps(ordered)
    caller_lat_pos = [gap for gap in caller_lat if gap >= 0]

    overlap_total = 0.0
    overlap_count = 0
    for c in caller:
        for a in agent:
            ov = _overlap(c, a)
            if ov > 0:
                overlap_total += ov
                overlap_count += 1

    c_dur = _summary(caller_dur)
    a_dur = _summary(agent_dur)
    c_lat = _summary(caller_lat)
    a_lat = _summary(agent_lat)
    c_pos = _summary(caller_lat_pos)

    return {
        "caller_talk_ratio": float(caller_speech / duration_s),
        "agent_talk_ratio": float(agent_speech / duration_s),
        "overlap_ratio": float(overlap_total / duration_s),
        "silence_ratio": float(
            max(0.0, 1.0 - (caller_speech + agent_speech - overlap_total) / duration_s)
        ),
        "caller_segment_count": len(caller),
        "agent_segment_count": len(agent),
        "overlap_event_count": overlap_count,
        "overlap_total_s": float(overlap_total),
        "caller_utterance_mean_s": c_dur["mean"],
        "caller_utterance_std_s": c_dur["std"],
        "caller_utterance_median_s": c_dur["median"],
        "caller_utterance_cv": c_dur["cv"],
        "agent_utterance_mean_s": a_dur["mean"],
        "agent_utterance_std_s": a_dur["std"],
        "agent_utterance_median_s": a_dur["median"],
        "agent_utterance_cv": a_dur["cv"],
        "caller_response_latency_mean_s": c_lat["mean"],
        "caller_response_latency_std_s": c_lat["std"],
        "caller_response_latency_cv": c_lat["cv"],
        "caller_response_latency_median_s": c_lat["median"],
        "caller_response_latency_min_s": c_lat["min"],
        "caller_response_latency_max_s": c_lat["max"],
        "caller_response_latency_pos_mean_s": c_pos["mean"],
        "caller_response_latency_pos_median_s": c_pos["median"],
        "caller_response_latency_pos_min_s": c_pos["min"],
        "caller_response_latency_pos_max_s": c_pos["max"],
        "caller_response_latency_pos_cv": c_pos["cv"],
        "agent_response_latency_mean_s": a_lat["mean"],
        "agent_response_latency_std_s": a_lat["std"],
        "agent_response_latency_cv": a_lat["cv"],
        "turn_ratio": float(len(caller) / max(len(agent), 1)),
        "caller_vs_agent_speech": float(caller_speech / max(agent_speech, 1e-6)),
        "caller_turns_per_min": float(len(caller) / (duration_s / 60.0)),
        "caller_sec_per_turn": float(caller_speech / max(len(caller), 1)),
    }
