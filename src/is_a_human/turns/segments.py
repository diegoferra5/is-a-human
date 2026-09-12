"""VAD segment types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VadSegment:
    channel: int
    start: float
    end: float
