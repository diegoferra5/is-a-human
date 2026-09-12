"""Behavioral view: features from turn-timing alone (turns.json).

No audio, no transcript, no GPU. These features describe *how the caller
behaves in the conversation* -- response latency, turn regularity, overlaps,
talk ratio -- which is speaker-independent and therefore our best bet for
generalizing to the unseen callers/voices in the hidden judging set.

Each call -> one flat dict of float features (a feature vector).
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median, pstdev

from src.config import CALLER_CH, AGENT_CH


def _load_turns(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f).get("turns", [])


def _stats(values: list[float], prefix: str) -> dict:
    """Robust summary stats; safe on empty/singleton lists."""
    if not values:
        return {f"{prefix}_{k}": 0.0 for k in
                ("mean", "std", "median", "min", "max", "cv", "n")}
    m = mean(values)
    sd = pstdev(values) if len(values) > 1 else 0.0
    return {
        f"{prefix}_mean": float(m),
        f"{prefix}_std": float(sd),
        f"{prefix}_median": float(median(values)),
        f"{prefix}_min": float(min(values)),
        f"{prefix}_max": float(max(values)),
        f"{prefix}_cv": float(sd / m) if m else 0.0,   # coeff. of variation: regularity
        f"{prefix}_n": float(len(values)),
    }


def _overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def extract(turns_path: Path, duration_s: float | None = None) -> dict:
    turns = _load_turns(turns_path)
    caller = sorted((t for t in turns if t["channel"] == CALLER_CH),
                    key=lambda t: t["start"])
    agent = sorted((t for t in turns if t["channel"] == AGENT_CH),
                   key=lambda t: t["start"])
    ordered = sorted(turns, key=lambda t: t["start"])

    caller_dur = [t["end"] - t["start"] for t in caller]
    agent_dur = [t["end"] - t["start"] for t in agent]
    caller_speech = sum(caller_dur)
    agent_speech = sum(agent_dur)

    if duration_s is None:
        duration_s = max((t["end"] for t in turns), default=0.0)
    duration_s = max(duration_s, 1e-6)

    caller_latencies: list[float] = []
    agent_latencies: list[float] = []
    prev = None
    for t in ordered:
        if prev is not None and prev["channel"] != t["channel"]:
            gap = t["start"] - prev["end"]   # negative = overlap / barge-in
            if t["channel"] == CALLER_CH:
                caller_latencies.append(gap)
            else:
                agent_latencies.append(gap)
        prev = t
        
    overlap_total = 0.0
    overlap_count = 0
    j = 0
    for c in caller:
        for a in agent:
            ov = _overlap(c, a)
            if ov > 0:
                overlap_total += ov
                overlap_count += 1

    feats: dict[str, float] = {}
    feats.update(_stats(caller_dur, "caller_dur"))
    feats.update(_stats(agent_dur, "agent_dur"))
    feats.update(_stats(caller_latencies, "resp_lat"))       # <- key signal
    feats.update(_stats(agent_latencies, "agent_lat"))
    feats.update(_stats([g for g in caller_latencies if g >= 0], "resp_lat_pos"))

    feats.update({
        "duration_s": float(duration_s),
        "n_caller_turns": float(len(caller)),
        "n_agent_turns": float(len(agent)),
        "turn_ratio": float(len(caller) / max(len(agent), 1)),
        "caller_speech_s": float(caller_speech),
        "agent_speech_s": float(agent_speech),
        "caller_speech_ratio": float(caller_speech / duration_s),
        "agent_speech_ratio": float(agent_speech / duration_s),
        "caller_vs_agent_speech": float(caller_speech / max(agent_speech, 1e-6)),
        "caller_turns_per_min": float(len(caller) / (duration_s / 60.0)),
        "silence_ratio": float(max(0.0, 1.0 - (caller_speech + agent_speech - overlap_total) / duration_s)),
        "overlap_total_s": float(overlap_total),
        "overlap_count": float(overlap_count),
        "overlap_ratio": float(overlap_total / duration_s),
        "bargein_count": float(sum(1 for g in caller_latencies if g < 0)),
        "avg_words_proxy": float(caller_speech / max(len(caller), 1)),  # sec/turn as a length proxy
    })
    return feats


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1])
    for k, v in extract(p).items():
        print(f"{k:28s} {v:.4f}")
