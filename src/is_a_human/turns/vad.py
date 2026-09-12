"""Dual-channel voice activity detection."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from is_a_human.turns.backends import (
    DEFAULT_VAD_BACKEND,
    VadBackend,
    VadBackendName,
    available_backends,
    get_backend,
)
from is_a_human.turns.segments import VadSegment
from is_a_human.turns.silero_vad import get_vad_model

__all__ = [
    "DEFAULT_VAD_BACKEND",
    "VadBackend",
    "VadBackendName",
    "VadSegment",
    "available_backends",
    "detect_dual_channel_segments",
    "detect_speech_segments",
    "get_backend",
    "get_vad_model",
    "segments_to_mask",
]

# Backward-compatible alias used by api/app.py
_get_vad_model = get_vad_model


def detect_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    channel: int,
    backend: VadBackend | VadBackendName | None = None,
    min_speech_duration_s: float = 0.12,
) -> list[VadSegment]:
    """Run VAD on a mono channel and return speech segments."""
    if audio.size == 0:
        return []

    resolved = backend if isinstance(backend, VadBackend) else get_backend(
        backend if isinstance(backend, str) else None
    )

    if resolved.name == "silero" and min_speech_duration_s != 0.12:
        from is_a_human.turns.silero_vad import detect_silero_segments

        return detect_silero_segments(
            audio,
            sample_rate,
            channel=channel,
            min_speech_duration_s=min_speech_duration_s,
        )

    return resolved.detect_segments(audio, sample_rate, channel=channel)


def detect_dual_channel_segments(
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    *,
    backend: VadBackend | VadBackendName | None = None,
) -> list[VadSegment]:
    """Detect speech segments on both caller and agent channels."""
    resolved = backend if isinstance(backend, VadBackend) else get_backend(
        backend if isinstance(backend, str) else None
    )
    return resolved.detect_dual_channel(ch0_caller, ch1_agent, sample_rate)


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
