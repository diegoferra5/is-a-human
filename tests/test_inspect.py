import pytest

from is_a_human.eval.inspect import _error_kind, inspect_rows, render_inspect_markdown, summarize_scores
from tests.test_tandem import _toy_splits

from is_a_human.detect.tandem import resolve_disagreement, train_tandem_from_rows


def test_error_kind_splits_mixer_vs_agree():
    assert _error_kind("human", "human", "human", "human") == "ok"
    assert _error_kind("human", "human", "human", "synthetic") == "ok_disagree"
    assert _error_kind("human", "synthetic", "synthetic", "synthetic") == "miss_agree"
    assert _error_kind("synthetic", "human", "human", "synthetic") == "miss_mixer"


def test_unsure_disagreement_uses_sharper_head():
    # Mixer barely synthetic; acoustic is sure human. Override to acoustic.
    assert resolve_disagreement(0.551, 0.021, 0.947) == pytest.approx(0.021)
    # Mixer is sure; leave it even if acoustic is more extreme.
    assert resolve_disagreement(0.279, 0.846, 0.183) == pytest.approx(0.279)
    assert resolve_disagreement(0.356, 0.663, 0.368) == pytest.approx(0.356)
    # Mixer wrong but outside the unsure band — do not steal the sharper acoustic.
    assert resolve_disagreement(0.332, 0.870, 0.205) == pytest.approx(0.332)
    # Heads agree: never override.
    assert resolve_disagreement(0.80, 0.70, 0.90) == pytest.approx(0.80)


def test_inspect_toy_model_writes_miss_fields():
    train, val = _toy_splits()
    model = train_tandem_from_rows(train, val)
    payload = inspect_rows(model, val)
    summary = summarize_scores(payload["calls"])
    assert summary["calls"] == 2
    assert "accuracy" in summary
    assert "balanced_accuracy" in summary
    markdown = render_inspect_markdown(payload)
    assert "Tandem inspection" in markdown
    for row in payload["calls"]:
        assert "p_fused" in row
        assert "caller_response_latency_pos_median_s" in row["behavioral_features"]
        assert "caller_rms_cv" in row["acoustic_features"]
