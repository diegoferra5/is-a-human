"""Energy-based VAD ported from origin/max branch."""

from __future__ import annotations

import numpy as np

from is_a_human.turns.postprocess import mask_to_segments, resample_bool_mask

FRAME_MS = 25
HOP_MS = 10
# Fitted to organizer turn JSON (8 kHz telephony). 6 dB over-detected
# non-speech energy; 14 dB / 12th-percentile floor matches the labels.
THRESH_DB = 14.0
FLOOR_PERCENTILE = 12
ENERGY_FRAME_DURATION_S = HOP_MS / 1000.0


def frame_energy_db(sig: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    """Return per-frame energy in dB and hop size in samples."""
    frame = int(sample_rate * FRAME_MS / 1000)
    hop = int(sample_rate * HOP_MS / 1000)
    if len(sig) < frame:
        return np.array([]), float(hop)

    count = 1 + (len(sig) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    frames = sig[idx]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-10)
    return 20 * np.log10(rms + 1e-10), float(hop)


def energy_speech_mask(
    audio: np.ndarray,
    sample_rate: int,
    *,
    thresh_db: float = THRESH_DB,
    floor_percentile: float = FLOOR_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frame_start_times, boolean_speech_mask) for energy VAD."""
    db, hop = frame_energy_db(audio, sample_rate)
    if db.size == 0:
        return np.array([]), np.array([], dtype=bool)

    floor = float(np.percentile(db, floor_percentile))
    mask = db > (floor + thresh_db)
    hop_s = hop / sample_rate
    times = np.arange(db.size, dtype=np.float64) * hop_s
    return times, mask


def energy_mask_at_times(
    audio: np.ndarray,
    sample_rate: int,
    target_times: np.ndarray,
    *,
    thresh_db: float = THRESH_DB,
) -> np.ndarray:
    """Resample energy VAD onto an arbitrary frame grid."""
    times, mask = energy_speech_mask(audio, sample_rate, thresh_db=thresh_db)
    if times.size == 0:
        return np.zeros(target_times.size, dtype=bool)
    return resample_bool_mask(times, mask, target_times)


def detect_energy_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    channel: int,
    frame_duration_s: float = ENERGY_FRAME_DURATION_S,
) -> list:
    """Run energy VAD and return speech segments."""
    if audio.size == 0:
        return []

    times, mask = energy_speech_mask(audio, sample_rate)
    if times.size == 0:
        return []

    return mask_to_segments(
        mask,
        frame_duration_s=frame_duration_s,
        channel=channel,
    )
