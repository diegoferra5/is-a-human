"""Dual-channel voice activity detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from silero_vad import get_speech_timestamps

_vad_model = None


@dataclass(frozen=True)
class VadSegment:
    channel: int
    start: float
    end: float


def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        _vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
    return _vad_model


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    channel: int,
    min_speech_duration_s: float = 0.2,
) -> list[VadSegment]:
    """Run Silero VAD on a mono channel and return speech segments."""
    if audio.size == 0:
        return []

    model = _get_vad_model()
    tensor = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=int(min_speech_duration_s * 1000),
        return_seconds=True,
    )
    return [
        VadSegment(channel=channel, start=float(ts["start"]), end=float(ts["end"]))
        for ts in timestamps
    ]


def detect_dual_channel_segments(
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
) -> list[VadSegment]:
    """Detect speech segments on both caller and agent channels."""
    caller_segments = detect_speech_segments(ch0_caller, sample_rate, channel=0)
    agent_segments = detect_speech_segments(ch1_agent, sample_rate, channel=1)
    return caller_segments + agent_segments


def segments_to_mask(
    segments: Sequence[VadSegment],
    duration_s: float,
    sample_rate: int,
    frame_duration_s: float = 0.032,
) -> np.ndarray:
    """Convert segments for one channel into a boolean frame mask."""
    num_frames = max(1, int(np.ceil(duration_s / frame_duration_s)))
    mask = np.zeros(num_frames, dtype=bool)
    for segment in segments:
        start_frame = int(segment.start / frame_duration_s)
        end_frame = int(np.ceil(segment.end / frame_duration_s))
        mask[start_frame:end_frame] = True
    return mask
