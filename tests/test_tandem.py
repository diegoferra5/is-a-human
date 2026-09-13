import numpy as np
import pytest

from is_a_human.analysis.features import BEHAVIORAL_HEAD_FEATURES, CallFeatures, extract_call_features
from is_a_human.detect.tandem import load_tandem, save_tandem, train_tandem_from_rows
from is_a_human.pipeline import PipelineResult
from is_a_human.turns.ledger import TurnEvent, TurnLedger, TurnType
from is_a_human.turns.metrics import ConversationMetrics
from is_a_human.turns.segments import VadSegment


def _row(label: str, talk: float, agent: float, rms: float) -> CallFeatures:
    base = {name: 0.0 for name in CallFeatures.numeric_field_names()}
    zcr = 0.85 if label == "human" else 0.12
    base.update(
        caller_talk_ratio=talk,
        agent_talk_ratio=agent,
        caller_rms_mean=rms,
        caller_zcr_std=zcr,
        caller_rms_cv=zcr,
        caller_crest_factor_cv=zcr,
        caller_spectral_flatness_std=zcr,
        caller_spectral_centroid_std=zcr * 100,
        caller_response_latency_pos_median_s=0.2 if label == "human" else 0.9,
        agent_aligned_recovery_cv=0.9 if label == "human" else 0.4,
    )
    return CallFeatures(anon_id=f"{label}_{talk}_{rms}", label=label, split="train", **base)


def _toy_splits():
    train = [
        _row("human", 0.15, 0.55, 0.05),
        _row("human", 0.16, 0.57, 0.06),
        _row("human", 0.14, 0.56, 0.04),
        _row("synthetic", 0.40, 0.30, 0.40),
        _row("synthetic", 0.42, 0.28, 0.38),
        _row("synthetic", 0.38, 0.32, 0.42),
    ]
    val = [
        _row("human", 0.17, 0.54, 0.07),
        _row("synthetic", 0.41, 0.29, 0.39),
    ]
    return train, val


def test_extract_call_features_from_vad_ledger():
    sample_rate = 8000
    duration_s = 2.0
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate
    ch0 = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    ch1 = (0.2 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    segments = (
        VadSegment(channel=0, start=0.0, end=0.8),
        VadSegment(channel=1, start=0.9, end=1.2),
        VadSegment(channel=0, start=1.3, end=2.0),
    )
    ledger = TurnLedger(
        events=(
            TurnEvent(channel=0, start=0.0, end=0.8, type=TurnType.SPEECH),
            TurnEvent(channel=None, start=0.8, end=0.9, type=TurnType.SILENCE),
            TurnEvent(channel=1, start=0.9, end=1.2, type=TurnType.SPEECH),
            TurnEvent(channel=None, start=1.2, end=1.3, type=TurnType.SILENCE),
            TurnEvent(channel=0, start=1.3, end=2.0, type=TurnType.SPEECH),
        ),
        speech_segments=segments,
        frame_duration_s=0.032,
    )
    metrics = ConversationMetrics(
        caller_talk_time_s=1.5,
        agent_talk_time_s=0.3,
        overlap_time_s=0.0,
        silence_time_s=0.2,
        interruption_count=0,
        mean_agent_response_latency_s=0.1,
        mean_caller_response_latency_s=0.1,
    )
    result = PipelineResult(ledger=ledger, metrics=metrics, latency_ms=1.0)
    features = extract_call_features(
        anon_id="toy",
        label="human",
        split="val",
        ch0_caller=ch0,
        ch1_agent=ch1,
        sample_rate=sample_rate,
        pipeline_result=result,
    )
    assert features.caller_rms_mean > 0
    assert features.caller_talk_ratio == pytest.approx(0.75)
    assert features.agent_aligned_recovery_count >= 1.0


def test_train_tandem_separates_and_roundtrips(tmp_path):
    train, val = _toy_splits()
    model = train_tandem_from_rows(train, val)
    assert model.fusion_type in {"stacked", "concatenated"}
    path = tmp_path / "tandem.json"
    save_tandem(model, path)
    loaded = load_tandem(path)
    assert loaded.disagree_unsure == pytest.approx(model.disagree_unsure)
    assert loaded.talkative_turn_min == model.talkative_turn_min
    probability, views = loaded.predict(val[1])
    assert 0.0 <= probability <= 1.0
    assert probability > 0.5
    assert set(views) == {"acoustic", "behavioral"}


def test_default_heads_drop_rms_mean_and_collinear_latency():
    train, val = _toy_splits()
    model = train_tandem_from_rows(train, val)
    assert "caller_rms_mean" not in model.acoustic.feature_names
    assert model.behavioral.feature_names == BEHAVIORAL_HEAD_FEATURES
    assert "caller_response_latency_pos_mean_s" not in model.behavioral.feature_names


def _with_quiet_patient(row: CallFeatures, *, turns: int) -> CallFeatures:
    values = {name: getattr(row, name) for name in CallFeatures.numeric_field_names()}
    values.update(
        caller_response_latency_pos_median_s=1.51,
        caller_rms_cv=0.28,
        caller_segment_count=turns,
    )
    return CallFeatures(anon_id=row.anon_id, label=row.label, split=row.split, **values)


def test_talkative_quiet_predict_flips_only_many_turns():
    train, val = _toy_splits()
    model = train_tandem_from_rows(train, val)
    sparse = _with_quiet_patient(val[1], turns=10)
    talkative = _with_quiet_patient(val[1], turns=24)
    p_sparse, _ = model.predict(sparse)
    p_talkative, _ = model.predict(talkative)
    assert p_sparse >= 0.5
    assert p_talkative < 0.5
    assert p_talkative == pytest.approx(1.0 - p_sparse)
