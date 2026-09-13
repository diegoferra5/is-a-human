"""Graceful degradation: masked heads, VAD failure, forced semantic path."""

from dataclasses import replace

import numpy as np
import pytest

from is_a_human.analysis.features import CallFeatures
from is_a_human.analysis.semantic import SEMANTIC_ZEROS
from is_a_human.detect.tandem import DEFAULT_MODEL_PATH, load_tandem


@pytest.fixture(scope="module")
def model():
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("models/tandem.json not trained")
    return load_tandem(DEFAULT_MODEL_PATH)


def _row(**over) -> CallFeatures:
    names = CallFeatures.numeric_field_names()
    base = {n: 0.0 for n in names}
    base.update(anon_id="x", label="unknown", split="live", duration_s=120.0,
                caller_segment_count=12, agent_segment_count=15, caller_talk_ratio=0.35,
                caller_response_latency_pos_median_s=1.2, agent_talk_ratio=0.5,
                agent_aligned_recovery_cv=0.8, caller_rms_cv=0.3, caller_zcr_std=0.04,
                caller_crest_factor_cv=0.2, caller_spectral_flatness_std=0.01,
                caller_spectral_centroid_std=150.0)
    base.update(over)
    return CallFeatures(**base)


def test_vad_failure_masks_fast_heads_and_forces_semantic(model):
    if not model.has_semantic():
        pytest.skip("no semantic head in model")
    dead = _row(caller_segment_count=0, agent_segment_count=0, caller_talk_ratio=0.0)
    p, views = model.predict(dead)
    assert views["acoustic_ok"] == 0.0 and views["behavioral_ok"] == 0.0
    assert views["acoustic"] == pytest.approx(model.neutral("acoustic"))
    assert model.needs_semantic(p, model.head_availability(dead))
    assert 0.0 <= p <= 1.0


def test_masked_head_contributes_nothing_to_fast_fusion(model):
    live = _row()
    p_live, _ = model.predict(live)
    # same row but the acoustic head unavailable: the fusion sees the neutral value
    masked = _row(caller_talk_ratio=0.0)
    p_masked, views = model.predict(masked)
    assert views["acoustic_ok"] == 0.0 and views["behavioral_ok"] == 1.0
    assert p_masked != p_live
    assert 0.0 <= p_masked <= 1.0


def test_predict_full_falls_back_without_transcript(model):
    row = _row()
    assert model.predict_full(row)[0] == pytest.approx(model.predict(row)[0])


def test_predict_full_uses_semantic_when_available(model):
    if not model.has_semantic():
        pytest.skip("no semantic head in model")
    sem = {k: 0.0 for k in SEMANTIC_ZEROS}
    sem.update(semantic_available=1.0, sem_speech_rate=2.9, sem_closing_ritual=1.0,
               sem_calls_agent_by_name=1.0, sem_positional_correction=1.0, sem_median_turn_words=7.0)
    row = replace(_row(), **sem)
    p3, views = model.predict_full(row)
    assert "semantic" in views and 0.0 <= p3 <= 1.0


def test_disagreement_opens_the_gate(model):
    if not model.has_semantic():
        pytest.skip("no semantic head in model")
    views = {"acoustic": 0.85, "behavioral": 0.20, "acoustic_ok": 1.0, "behavioral_ok": 1.0}
    assert model.heads_disagree(views)
    assert model.needs_semantic(0.9, {"acoustic": True, "behavioral": True}, views)
    agree = {"acoustic": 0.85, "behavioral": 0.80, "acoustic_ok": 1.0, "behavioral_ok": 1.0}
    assert not model.needs_semantic(0.9, {"acoustic": True, "behavioral": True}, agree)


def test_silence_through_live_path_does_not_crash(model):
    from is_a_human.detect.live import detect_from_audio
    sr = 8000
    silence = np.zeros(sr * 20, dtype=np.float32)
    calls = []

    def fake_transcribe(ch0, sample_rate, segments):
        calls.append(segments)
        return [{"channel": 0, "start": 0.0, "end": 1.0, "text": "Sí."}]

    p, views, timings = detect_from_audio(model, silence, silence, sr, transcribe=fake_transcribe)
    assert 0.0 <= p <= 1.0
    if model.has_semantic():
        assert calls and calls[0] == [(0.0, 20.0)]      # whole channel when the VAD found nothing
        assert timings["gated"] == 1.0
