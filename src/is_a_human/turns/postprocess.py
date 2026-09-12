"""Shared VAD post-processing: boolean frame masks to speech segments."""

from __future__ import annotations

import numpy as np

from is_a_human.turns.segments import VadSegment

MIN_SPEECH_MS = 120
MIN_GAP_MS = 280


def mask_to_segments(
    mask: np.ndarray,
    *,
    frame_duration_s: float,
    channel: int,
    min_speech_s: float = MIN_SPEECH_MS / 1000.0,
    min_gap_s: float = MIN_GAP_MS / 1000.0,
) -> list[VadSegment]:
    """Convert a boolean speech mask into merged time segments."""
    if mask.size == 0:
        return []

    raw: list[tuple[float, float]] = []
    in_seg = False
    start = 0.0

    for idx, active in enumerate(mask):
        t = idx * frame_duration_s
        if active and not in_seg:
            start = t
            in_seg = True
        elif not active and in_seg:
            raw.append((start, t + frame_duration_s))
            in_seg = False

    if in_seg:
        raw.append((start, mask.size * frame_duration_s))

    merged: list[tuple[float, float]] = []
    for seg_start, seg_end in raw:
        if merged and seg_start - merged[-1][1] < min_gap_s:
            merged[-1] = (merged[-1][0], seg_end)
        else:
            merged.append((seg_start, seg_end))

    return [
        VadSegment(channel=channel, start=float(seg_start), end=float(seg_end))
        for seg_start, seg_end in merged
        if (seg_end - seg_start) >= min_speech_s
    ]


def resample_bool_mask(
    source_times: np.ndarray,
    source_mask: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Sample a boolean mask onto a new time grid (True if any source frame overlaps)."""
    if source_times.size == 0:
        return np.zeros(target_times.size, dtype=bool)

    hop_s = float(np.median(np.diff(source_times))) if source_times.size > 1 else 0.01
    frame_s = hop_s
    result = np.zeros(target_times.size, dtype=bool)

    for idx, target_t in enumerate(target_times):
        window_end = target_t + frame_s
        overlap = (source_times < window_end) & (source_times + frame_s > target_t)
        if np.any(overlap):
            result[idx] = bool(np.any(source_mask[overlap]))

    return result
