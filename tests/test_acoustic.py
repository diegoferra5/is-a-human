"""Acoustic descriptors on caller-channel speech segments."""

import numpy as np
import pytest

from is_a_human.analysis.acoustic import extract_acoustic_features
from is_a_human.dataset.loader import TurnSegment

pytestmark = pytest.mark.acoustic


def test_extract_acoustic_features_on_tone(report_metrics):
    sample_rate = 8000
    duration_s = 1.0
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    ch0 = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    turns = (TurnSegment(channel=0, start=0.1, end=0.9),)

    features = extract_acoustic_features(ch0, sample_rate, turns, duration_s)
    report_metrics(
        caller_rms_mean=features["caller_rms_mean"],
        caller_zcr_mean=features["caller_zcr_mean"],
        caller_spectral_centroid_mean=features["caller_spectral_centroid_mean"],
        caller_crest_factor_mean=features["caller_crest_factor_mean"],
    )

    assert features["caller_rms_mean"] > 0.2
    assert features["caller_zcr_mean"] > 0.0
    assert features["caller_spectral_centroid_mean"] > 0.0
    assert features["caller_crest_factor_mean"] > 0.0


def test_silent_caller_segments_are_zero():
    sample_rate = 8000
    ch0 = np.zeros(sample_rate, dtype=np.float32)
    turns = (TurnSegment(channel=0, start=0.1, end=0.5),)
    features = extract_acoustic_features(ch0, sample_rate, turns, 1.0)

    assert features["caller_rms_mean"] == 0.0
    assert features["caller_zcr_mean"] == 0.0
