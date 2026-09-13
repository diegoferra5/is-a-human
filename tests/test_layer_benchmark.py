from is_a_human.analysis.features import CallFeatures
from is_a_human.eval.benchmark_html import render_benchmark_page
from is_a_human.eval.layer_benchmark import (
    ACOUSTIC_FEATURES,
    BEHAVIORAL_FEATURES,
    fit_and_score,
    run_layer_benchmarks,
    select_top_features,
)


def test_feature_groups_partition_numeric_fields():
    numeric = set(CallFeatures.numeric_field_names()) - {"duration_s"}
    assert "caller_rms_mean" in ACOUSTIC_FEATURES
    assert "caller_talk_ratio" in BEHAVIORAL_FEATURES
    assert set(ACOUSTIC_FEATURES).isdisjoint(BEHAVIORAL_FEATURES)
    assert set(ACOUSTIC_FEATURES) | set(BEHAVIORAL_FEATURES) == numeric


def _row(label: str, caller_talk: float, agent_talk: float) -> CallFeatures:
    base = {name: 0.0 for name in CallFeatures.numeric_field_names()}
    base.update({"caller_talk_ratio": caller_talk, "agent_talk_ratio": agent_talk})
    return CallFeatures(anon_id=f"id_{label}_{caller_talk}", label=label, split="train", **base)


def test_select_top_features_ranks_separating_signal():
    rows = [
        _row("human", 0.15, 0.55),
        _row("human", 0.16, 0.57),
        _row("synthetic", 0.40, 0.30),
        _row("synthetic", 0.42, 0.28),
    ]
    ranked = select_top_features(rows, ("caller_talk_ratio", "agent_talk_ratio"), k=2)
    assert ranked[0]["feature"] == "caller_talk_ratio"
    assert abs(ranked[0]["effect_size"]) > abs(ranked[1]["effect_size"])


def test_fit_and_score_reports_val_auc():
    train = [
        _row("human", 0.15, 0.55),
        _row("human", 0.16, 0.57),
        _row("human", 0.14, 0.56),
        _row("synthetic", 0.40, 0.30),
        _row("synthetic", 0.42, 0.28),
        _row("synthetic", 0.38, 0.32),
    ]
    val = [
        _row("human", 0.17, 0.54),
        _row("synthetic", 0.41, 0.29),
    ]
    scored = fit_and_score(train, val, BEHAVIORAL_FEATURES, k=2)
    assert scored is not None
    assert scored["val"]["auc"] >= 0.8
    assert scored["features_used"]


def test_missing_dataset_skips_suites():
    payload = run_layer_benchmarks(
        dataset_root="/nonexistent/path",
        suites=("vad", "semantic", "tandem"),
        show_progress=False,
    )
    by_id = {suite["id"]: suite for suite in payload["suites"]}
    assert by_id["vad"]["status"] == "skipped"
    assert by_id["semantic"]["status"] == "skipped"
    assert by_id["tandem"]["status"] == "skipped"


def test_render_benchmark_page_includes_suites():
    html = render_benchmark_page(
        {
            "generated_at": "2026-09-12 22:00 UTC",
            "duration_s": 1.2,
            "split": "val",
            "limit": 8,
            "suites": [
                {
                    "id": "vad",
                    "title": "VAD",
                    "purpose": "segments",
                    "status": "ok",
                    "headline": "0.900 overall IoU",
                    "headline_detail": "hybrid · 8 calls",
                    "reason": "",
                    "table": {
                        "columns": ["backend", "overall"],
                        "rows": [["hybrid", 0.9]],
                    },
                    "top_separators": [],
                },
                {
                    "id": "semantic",
                    "title": "Semantic",
                    "purpose": "traps",
                    "status": "skipped",
                    "headline": "skipped",
                    "headline_detail": "no transcripts",
                    "reason": "no transcripts",
                    "table": None,
                    "top_separators": [],
                },
            ],
        }
    )
    assert "Layer benchmarks" in html
    assert "0.900 overall IoU" in html
    assert "no transcripts" in html


def test_fit_and_score_none_without_both_classes():
    train = [_row("human", 0.1, 0.5) for _ in range(4)]
    val = [_row("human", 0.1, 0.5), _row("human", 0.2, 0.4)]
    assert fit_and_score(train, val, ("caller_talk_ratio",), k=1) is None
