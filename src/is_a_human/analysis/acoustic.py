"""Unconventional acoustic features on caller channel segments."""

from __future__ import annotations

import numpy as np

from typing import Protocol


class _SpeechSegment(Protocol):
    channel: int
    start: float
    end: float


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


def _zero_crossing_rate(frame: np.ndarray) -> float:
    if frame.size < 2:
        return 0.0
    signs = np.sign(frame)
    return float(np.mean(signs[:-1] != signs[1:]))


def _spectral_features(frame: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    if frame.size < 16:
        return 0.0, 0.0, 0.0

    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sample_rate)
    power = spectrum**2
    total = float(np.sum(power))
    if total <= 0:
        return 0.0, 0.0, 0.0

    centroid = float(np.sum(freqs * power) / total)
    geo_mean = float(np.exp(np.mean(np.log(power + 1e-12))))
    arith_mean = float(np.mean(power))
    flatness = geo_mean / arith_mean if arith_mean > 0 else 0.0

    hf_mask = freqs >= 1000.0
    lf_mask = freqs < 1000.0
    hf_energy = float(np.sum(power[hf_mask]))
    lf_energy = float(np.sum(power[lf_mask]))
    hf_lf_ratio = hf_energy / lf_energy if lf_energy > 0 else 0.0

    return centroid, flatness, hf_lf_ratio


def _crest_factor(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(frame**2)))
    peak = float(np.max(np.abs(frame)))
    return peak / rms if rms > 0 else 0.0


def _silence_gaps(segments: list[_SpeechSegment], duration_s: float) -> list[float]:
    channel_segments = sorted(
        [(segment.start, segment.end) for segment in segments if segment.channel == 0],
        key=lambda item: item[0],
    )
    if not channel_segments:
        return []

    gaps: list[float] = []
    if channel_segments[0][0] > 0:
        gaps.append(channel_segments[0][0])

    for (_, prev_end), (next_start, _) in zip(channel_segments, channel_segments[1:]):
        gap = next_start - prev_end
        if gap > 0:
            gaps.append(gap)

    last_end = channel_segments[-1][1]
    if duration_s > last_end:
        gaps.append(duration_s - last_end)

    return gaps


def extract_acoustic_features(
    ch0_caller: np.ndarray,
    sample_rate: int,
    speech_segments: tuple[_SpeechSegment, ...],
    duration_s: float,
) -> dict[str, float]:
    """Extract unconventional acoustic descriptors from caller speech segments."""
    caller_segments = [segment for segment in speech_segments if segment.channel == 0]

    zcr_values: list[float] = []
    centroid_values: list[float] = []
    flatness_values: list[float] = []
    hf_lf_values: list[float] = []
    crest_values: list[float] = []
    rms_values: list[float] = []
    segment_lengths: list[float] = []

    for segment in caller_segments:
        frame = _frame_audio(ch0_caller, segment.start, segment.end, sample_rate)
        if frame.size < 8:
            continue
        segment_lengths.append(segment.end - segment.start)
        rms_values.append(float(np.sqrt(np.mean(frame**2))))
        zcr_values.append(_zero_crossing_rate(frame))
        centroid, flatness, hf_lf = _spectral_features(frame, sample_rate)
        centroid_values.append(centroid)
        flatness_values.append(flatness)
        hf_lf_values.append(hf_lf)
        crest_values.append(_crest_factor(frame))

    gap_mean, gap_std, gap_cv = _safe_stats(_silence_gaps(list(speech_segments), duration_s))
    seg_len_mean, seg_len_std, seg_len_cv = _safe_stats(segment_lengths)
    rms_mean, rms_std, rms_cv = _safe_stats(rms_values)
    zcr_mean, zcr_std, _ = _safe_stats(zcr_values)
    centroid_mean, centroid_std, _ = _safe_stats(centroid_values)
    flatness_mean, flatness_std, _ = _safe_stats(flatness_values)
    hf_lf_mean, hf_lf_std, _ = _safe_stats(hf_lf_values)
    crest_mean, crest_std, crest_cv = _safe_stats(crest_values)

    return {
        "caller_rms_mean": rms_mean,
        "caller_rms_std": rms_std,
        "caller_rms_cv": rms_cv,
        "caller_zcr_mean": zcr_mean,
        "caller_zcr_std": zcr_std,
        "caller_spectral_centroid_mean": centroid_mean,
        "caller_spectral_centroid_std": centroid_std,
        "caller_spectral_flatness_mean": flatness_mean,
        "caller_spectral_flatness_std": flatness_std,
        "caller_hf_lf_ratio_mean": hf_lf_mean,
        "caller_hf_lf_ratio_std": hf_lf_std,
        "caller_crest_factor_mean": crest_mean,
        "caller_crest_factor_std": crest_std,
        "caller_crest_factor_cv": crest_cv,
        "caller_intra_silence_gap_mean_s": gap_mean,
        "caller_intra_silence_gap_cv": gap_cv,
        "caller_segment_length_cv": seg_len_cv,
    }
