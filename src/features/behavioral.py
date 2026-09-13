"""Behavioral view: features from turn-timing alone (turns.json).

Thin wrapper around is_a_human.analysis.behavioral so Max's train/serve path
and Roger's live extractor share one implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

from is_a_human.analysis.behavioral import extract_timing_features
from is_a_human.dataset.loader import TurnSegment


def _load_turns(path: Path) -> tuple[TurnSegment, ...]:
    with open(path) as f:
        raw = json.load(f).get("turns", [])
    return tuple(
        TurnSegment(channel=int(t["channel"]), start=float(t["start"]), end=float(t["end"]))
        for t in raw
    )


def extract(turns_path: Path, duration_s: float | None = None) -> dict:
    turns = _load_turns(turns_path)
    if duration_s is None:
        duration_s = max((t.end for t in turns), default=0.0)
    return extract_timing_features(turns, duration_s)


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1])
    for k, v in extract(p).items():
        print(f"{k:40s} {v:.4f}")
