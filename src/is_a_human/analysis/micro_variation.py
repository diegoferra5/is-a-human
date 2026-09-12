"""Micro-variation features: formants, pitch jitter/shimmer, pause entropy, LFCC."""

from __future__ import annotations

import numpy as np

from is_a_human.analysis.dsp import (
    delta_features,
    estimate_f0,
    frame_rms,
    frame_signal,
    harmonic_to_noise_ratio,
    lfcc_coefficients,
    lpc_formants,
    normalized_sequence_jitter,
    normalized_sequence_shimmer,
    shannon_entropy,
)
from is_a_human.dataset.loader import TurnSegment


def _safe_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    cv = std / mean if mean > 0 else 0.0
    return mean, std, cv


def _frame_audio(audio: np.ndarray, start_s: float, end_s: float, sample_rate: int) -> np.ndarray:
    start_idx = max(0, int(start_s * sample_rate))
    end_idx = min(len(audio), int(end_s * sample_rate))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)
    return audio[start_idx:end_idx]


def _caller_silence_gaps(turns: tuple[TurnSegment, ...], duration_s: float) -> list[float]:
    segments = sorted(
        [(segment.start, segment.end) for segment in turns if segment.channel == 0],
        key=lambda item: item[0],
    )
    if not segments:
        return [duration_s] if duration_s > 0 else []

    gaps: list[float] = []
    if segments[0][0] > 0:
        gaps.append(segments[0][0])

    for (_, prev_end), (next_start, _) in zip(segments, segments[1:]):
        gap = next_start - prev_end
        if gap > 0:
            gaps.append(gap)

    last_end = segments[-1][1]
    if duration_s > last_end:
        gaps.append(duration_s - last_end)

    return gaps


def extract_micro_variation_features(
    ch0_caller: np.ndarray,
    sample_rate: int,
    organizer_turns: tuple[TurnSegment, ...],
    duration_s: float,
) -> dict[str, float]:
    """Formant volatility, pitch micro-tremors, pause entropy, LFCC delta-deltas."""
    caller_segments = [segment for segment in organizer_turns if segment.channel == 0]

    f1_track: list[float] = []
    f2_track: list[float] = []
    f3_track: list[float] = []
    f0_track: list[float] = []
    amp_track: list[float] = []
    hnr_track: list[float] = []
    lfcc_frames: list[np.ndarray] = []

    for segment in caller_segments:
        audio = _frame_audio(ch0_caller, segment.start, segment.end, sample_rate)
        for frame in frame_signal(audio, sample_rate, frame_ms=25.0, hop_ms=10.0):
            if frame.size < 32:
                continue

            hnr = harmonic_to_noise_ratio(frame, sample_rate)
            if hnr > 0:
                hnr_track.append(hnr)

            f1, f2, f3 = lpc_formants(frame, sample_rate)
            if f1 > 0:
                f1_track.append(f1)
            if f2 > 0:
                f2_track.append(f2)
            if f3 > 0:
                f3_track.append(f3)

            f0 = estimate_f0(frame, sample_rate)
            rms = frame_rms(frame)
            if f0 > 0:
                f0_track.append(f0)
            if rms > 0:
                amp_track.append(rms)

            lfcc_frames.append(lfcc_coefficients(frame, sample_rate))

    _, f1_std, _ = _safe_stats(f1_track)
    _, f2_std, _ = _safe_stats(f2_track)
    _, f3_std, _ = _safe_stats(f3_track)
    formant_volatility = float(np.mean([f1_std, f2_std, f3_std]))

    _, f0_std, f0_cv = _safe_stats(f0_track)
    hnr_mean, hnr_std, hnr_cv = _safe_stats(hnr_track)
    jitter = normalized_sequence_jitter(f0_track)
    shimmer = normalized_sequence_shimmer(amp_track)

    pause_entropy = shannon_entropy(_caller_silence_gaps(organizer_turns, duration_s))

    lfcc_dd_std = 0.0
    if len(lfcc_frames) >= 3:
        sequence = np.vstack(lfcc_frames)
        deltas = delta_features(sequence)
        dd = delta_features(deltas)
        lfcc_dd_std = float(np.std(dd))

    return {
        "caller_formant_f1_std": f1_std,
        "caller_formant_f2_std": f2_std,
        "caller_formant_f3_std": f3_std,
        "caller_formant_volatility_mean": formant_volatility,
        "caller_pitch_jitter": jitter,
        "caller_pitch_shimmer": shimmer,
        "caller_f0_std": f0_std,
        "caller_f0_cv": f0_cv,
        "caller_hnr_mean": hnr_mean,
        "caller_hnr_std": hnr_std,
        "caller_hnr_cv": hnr_cv,
        "caller_pause_entropy": pause_entropy,
        "caller_lfcc_delta_delta_std": lfcc_dd_std,
    }
