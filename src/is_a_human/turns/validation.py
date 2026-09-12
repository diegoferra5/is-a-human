"""Validation helpers for comparing VAD output to organizer turn files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from is_a_human.dataset.loader import TurnSegment
from is_a_human.turns.segments import VadSegment


@dataclass(frozen=True)
class SegmentIoU:
    channel: int
    intersection_s: float
    union_s: float

    @property
    def iou(self) -> float:
        if self.union_s <= 0:
            return 0.0
        return self.intersection_s / self.union_s


def _segments_to_intervals(segments: Sequence, channel: int) -> list[tuple[float, float]]:
    return [
        (segment.start, segment.end)
        for segment in segments
        if int(segment.channel) == channel
    ]


def _interval_union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def _interval_intersection_duration(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> float:
    total = 0.0
    for l_start, l_end in left:
        for r_start, r_end in right:
            start = max(l_start, r_start)
            end = min(l_end, r_end)
            if end > start:
                total += end - start
    return total


def compare_turn_segments(
    predicted: Sequence[VadSegment],
    reference: Sequence[TurnSegment],
    channel: int,
) -> SegmentIoU:
    """Compare predicted VAD segments to organizer-provided turns for one channel."""
    pred_intervals = _segments_to_intervals(predicted, channel)
    ref_intervals = _segments_to_intervals(reference, channel)

    intersection = _interval_intersection_duration(pred_intervals, ref_intervals)
    union = (
        _interval_union_duration(pred_intervals)
        + _interval_union_duration(ref_intervals)
        - intersection
    )
    return SegmentIoU(channel=channel, intersection_s=intersection, union_s=union)
