"""Lightweight DSP helpers for micro-variation and interaction features."""

from __future__ import annotations

import math

import numpy as np


def frame_signal(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> list[np.ndarray]:
    """Split audio into overlapping frames."""
    frame_len = max(16, int(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, int(sample_rate * hop_ms / 1000.0))
    if audio.size < frame_len:
        return [audio.astype(np.float64)] if audio.size else []

    frames: list[np.ndarray] = []
    for start in range(0, audio.size - frame_len + 1, hop_len):
        frames.append(audio[start : start + frame_len].astype(np.float64))
    return frames


def preemphasis(frame: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if frame.size < 2:
        return frame
    return np.append(frame[0], frame[1:] - coeff * frame[:-1])


def levinson_durbin(signal: np.ndarray, order: int) -> np.ndarray:
    """Return LPC coefficients including a0=1."""
    if signal.size == 0:
        return np.ones(order + 1)

    signal = signal - np.mean(signal)
    n = signal.size
    r = np.correlate(signal, signal, mode="full")[n - 1 : n + order]
    if r[0] <= 1e-12:
        return np.concatenate(([1.0], np.zeros(order)))

    a = np.zeros(order + 1)
    e = r[0]
    a[0] = 1.0

    for i in range(1, order + 1):
        if e <= 1e-12:
            break
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = (r[i] - acc) / e
        a_old = a.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = a_old[j] - k * a_old[i - j]
        e *= 1.0 - k * k

    return a


def lpc_formants(frame: np.ndarray, sample_rate: int, order: int = 12) -> tuple[float, float, float]:
    """Estimate F1/F2/F3 (Hz) from one frame via LPC root analysis."""
    if frame.size < order + 2:
        return 0.0, 0.0, 0.0

    windowed = preemphasis(frame) * np.hamming(frame.size)
    coeffs = levinson_durbin(windowed, order)
    roots = np.roots(coeffs)

    freqs: list[float] = []
    for root in roots:
        if np.abs(np.imag(root)) < 1e-3:
            continue
        freq = float(np.abs(np.arctan2(np.imag(root), np.real(root))) * sample_rate / (2 * math.pi))
        if 90.0 <= freq <= sample_rate / 2 - 50:
            freqs.append(freq)

    freqs.sort()
    if len(freqs) >= 3:
        return freqs[0], freqs[1], freqs[2]
    if len(freqs) == 2:
        return freqs[0], freqs[1], 0.0
    if len(freqs) == 1:
        return freqs[0], 0.0, 0.0
    return 0.0, 0.0, 0.0


def estimate_f0(frame: np.ndarray, sample_rate: int, *, fmin: float = 70.0, fmax: float = 400.0) -> float:
    """Autocorrelation-based F0 estimate for one frame."""
    if frame.size < 16:
        return 0.0

    frame = frame - np.mean(frame)
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[corr.size // 2 :]

    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(corr.size - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return 0.0

    segment = corr[min_lag : max_lag + 1]
    if segment.size == 0 or corr[0] <= 1e-12:
        return 0.0

    peak_idx = int(np.argmax(segment)) + min_lag
    if peak_idx <= 0:
        return 0.0
    return float(sample_rate / peak_idx)


def frame_rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame**2)))


def linear_filterbank_energies(frame: np.ndarray, sample_rate: int, n_bands: int = 22) -> np.ndarray:
    """Linear-spaced filterbank energies for LFCC."""
    if frame.size < 32:
        return np.zeros(n_bands)

    spectrum = np.abs(np.fft.rfft(frame)) ** 2
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sample_rate)
    max_freq = sample_rate / 2
    edges = np.linspace(0.0, max_freq, n_bands + 1)
    energies = np.zeros(n_bands)

    for band in range(n_bands):
        mask = (freqs >= edges[band]) & (freqs < edges[band + 1])
        if np.any(mask):
            energies[band] = float(np.sum(spectrum[mask]))

    return energies


def lfcc_coefficients(frame: np.ndarray, sample_rate: int, n_ceps: int = 13) -> np.ndarray:
    """Linear-frequency cepstral coefficients."""
    energies = linear_filterbank_energies(frame, sample_rate)
    log_energies = np.log(energies + 1e-12)
    cepstrum = np.fft.rfft(log_energies).real
    return cepstrum[:n_ceps]


def delta_features(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[0] < 2:
        return np.zeros_like(sequence)
    deltas = np.zeros_like(sequence)
    for t in range(sequence.shape[0]):
        prev_t = max(0, t - 1)
        next_t = min(sequence.shape[0] - 1, t + 1)
        deltas[t] = (sequence[next_t] - sequence[prev_t]) / 2.0
    return deltas


def shannon_entropy(values: list[float], *, n_bins: int = 8) -> float:
    """Shannon entropy (bits) of a positive-valued distribution."""
    if not values:
        return 0.0

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0

    counts, _ = np.histogram(arr, bins=n_bins)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0

    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def normalized_sequence_jitter(values: list[float]) -> float:
    """Relative frame-to-frame F0 variation."""
    clean = [v for v in values if v > 0]
    if len(clean) < 2:
        return 0.0
    diffs = [abs(clean[i] - clean[i - 1]) for i in range(1, len(clean))]
    mean_f0 = float(np.mean(clean))
    return float(np.mean(diffs) / mean_f0) if mean_f0 > 0 else 0.0


def harmonic_to_noise_ratio(frame: np.ndarray, sample_rate: int) -> float:
    """HNR (dB) from autocorrelation at the estimated pitch period."""
    f0 = estimate_f0(frame, sample_rate)
    if f0 <= 0 or frame.size < 64:
        return 0.0

    centered = frame - np.mean(frame)
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    if corr[0] <= 1e-12:
        return 0.0

    lag = int(round(sample_rate / f0))
    if lag <= 0 or lag >= corr.size:
        return 0.0

    peak = float(corr[lag] / corr[0])
    if peak <= 1e-6 or peak >= 0.999:
        return 0.0

    return float(10.0 * np.log10(peak / (1.0 - peak)))


def spectral_pink_slope(frame: np.ndarray, sample_rate: int) -> float:
    """Log-log spectral slope between 500 Hz and 3 kHz (negative ≈ pink noise)."""
    if frame.size < 64:
        return 0.0

    spectrum = np.abs(np.fft.rfft(frame)) ** 2
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sample_rate)
    mask = (freqs >= 500.0) & (freqs <= 3000.0)
    if int(np.sum(mask)) < 4:
        return 0.0

    log_f = np.log(freqs[mask])
    log_p = np.log(spectrum[mask] + 1e-12)
    return float(np.polyfit(log_f, log_p, 1)[0])


def normalized_sequence_shimmer(amplitudes: list[float]) -> float:
    """Relative frame-to-frame amplitude variation."""
    clean = [v for v in amplitudes if v > 0]
    if len(clean) < 2:
        return 0.0
    diffs = [abs(clean[i] - clean[i - 1]) for i in range(1, len(clean))]
    mean_amp = float(np.mean(clean))
    return float(np.mean(diffs) / mean_amp) if mean_amp > 0 else 0.0
