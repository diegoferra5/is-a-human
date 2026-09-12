"""Silero VAD with frame-level probability access."""

from __future__ import annotations

import numpy as np
import torch
from silero_vad import get_speech_timestamps

from is_a_human.turns.postprocess import mask_to_segments

_vad_model = None

SILERO_THRESHOLD = 0.5
# Exit speech only after probability falls this far (Silero hysteresis).
SILERO_NEG_THRESHOLD = 0.35
# Match organizer JSON: 10 ms resolution, 120 ms min speech, 280 ms gap.
SILERO_MIN_SPEECH_MS = 120
SILERO_MIN_SILENCE_MS = 280
SILERO_SPEECH_PAD_MS = 0
SILERO_TIME_RESOLUTION = 2


def _window_size_samples(sample_rate: int) -> int:
    return 512 if sample_rate == 16000 else 256


def frame_duration_s(sample_rate: int) -> float:
    return _window_size_samples(sample_rate) / sample_rate


def get_vad_model():
    """Load and cache the Silero VAD model."""
    global _vad_model
    if _vad_model is None:
        _vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
    return _vad_model


def iter_silero_probs(
    audio: np.ndarray,
    sample_rate: int,
    *,
    model=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_start_times, speech_probabilities) for each Silero chunk."""
    if audio.size == 0:
        return np.array([]), np.array([])

    model = model or get_vad_model()
    window = _window_size_samples(sample_rate)
    tensor = torch.from_numpy(audio).float()

    if hasattr(model, "reset_states"):
        model.reset_states()

    times: list[float] = []
    probs: list[float] = []

    for start in range(0, len(tensor), window):
        chunk = tensor[start : start + window]
        if chunk.numel() < window:
            break
        prob = float(model(chunk, sample_rate).item())
        times.append(start / sample_rate)
        probs.append(prob)

    return np.asarray(times, dtype=np.float64), np.asarray(probs, dtype=np.float64)


def silero_speech_mask(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = SILERO_THRESHOLD,
    model=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_start_times, boolean_speech_mask) from Silero probabilities."""
    times, probs = iter_silero_probs(audio, sample_rate, model=model)
    if times.size == 0:
        return times, np.array([], dtype=bool)
    return times, probs >= threshold


def detect_silero_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    channel: int,
    min_speech_duration_s: float = SILERO_MIN_SPEECH_MS / 1000.0,
    threshold: float = SILERO_THRESHOLD,
    model=None,
) -> list:
    """Run Silero VAD via get_speech_timestamps with telephony-tuned params."""
    from is_a_human.turns.segments import VadSegment

    if audio.size == 0:
        return []

    model = model or get_vad_model()
    tensor = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        neg_threshold=SILERO_NEG_THRESHOLD,
        min_speech_duration_ms=int(min_speech_duration_s * 1000),
        min_silence_duration_ms=SILERO_MIN_SILENCE_MS,
        speech_pad_ms=SILERO_SPEECH_PAD_MS,
        return_seconds=True,
        time_resolution=SILERO_TIME_RESOLUTION,
    )
    return [
        VadSegment(channel=channel, start=float(ts["start"]), end=float(ts["end"]))
        for ts in timestamps
    ]


def detect_silero_segments_from_mask(
    audio: np.ndarray,
    sample_rate: int,
    *,
    channel: int,
    threshold: float = SILERO_THRESHOLD,
    model=None,
) -> list:
    """Run Silero VAD with unified post-processing (for hybrid parity)."""
    times, mask = silero_speech_mask(audio, sample_rate, threshold=threshold, model=model)
    if times.size == 0:
        return []

    return mask_to_segments(
        mask,
        frame_duration_s=frame_duration_s(sample_rate),
        channel=channel,
    )
