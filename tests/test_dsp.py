import numpy as np

from is_a_human.analysis.dsp import (
    estimate_f0,
    harmonic_to_noise_ratio,
    lpc_formants,
    normalized_sequence_jitter,
    shannon_entropy,
)


def test_shannon_entropy_uniform_vs_peaked():
    uniform = shannon_entropy([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], n_bins=8)
    peaked = shannon_entropy([1.0, 1.01, 1.02, 8.0], n_bins=8)
    assert uniform > peaked


def test_pitch_jitter_increases_with_variation():
    stable = normalized_sequence_jitter([120.0, 120.5, 119.8, 120.2])
    volatile = normalized_sequence_jitter([120.0, 140.0, 110.0, 150.0])
    assert volatile > stable


def test_lpc_formants_on_synthetic_vowel():
    sample_rate = 8000
    t = np.arange(0, 0.1, 1 / sample_rate)
    frame = (0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)).astype(np.float64)
    f1, f2, f3 = lpc_formants(frame, sample_rate)
    assert f1 > 0 or f2 > 0


def test_estimate_f0_on_sine():
    sample_rate = 8000
    f0 = 150.0
    t = np.arange(0, 0.05, 1 / sample_rate)
    frame = np.sin(2 * np.pi * f0 * t)
    estimate = estimate_f0(frame, sample_rate)
    assert 120.0 <= estimate <= 180.0


def test_hnr_positive_on_harmonic_tone():
    sample_rate = 8000
    f0 = 150.0
    t = np.arange(0, 0.08, 1 / sample_rate)
    frame = np.sin(2 * np.pi * f0 * t)
    hnr = harmonic_to_noise_ratio(frame, sample_rate)
    assert hnr > 5.0
