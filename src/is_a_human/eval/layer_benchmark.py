"""Dataset benchmarks for VAD, acoustic, semantic, and behavioural layers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from types import SimpleNamespace
from typing import Iterable, Sequence

from is_a_human.analysis.classifier import evaluate_classifier, train_logistic_regression
from is_a_human.analysis.explore import _collect_features
from is_a_human.analysis.features import acoustic_feature_names, behavioral_feature_names
from is_a_human.analysis.semantic import FEATURE_NAMES as SEMANTIC_FEATURES, extract_semantic_features
from is_a_human.dataset.errors import DatasetError
from is_a_human.dataset.loader import iter_split
from is_a_human.eval.vad_benchmark import run_vad_benchmark_all

ACOUSTIC_FEATURES = acoustic_feature_names()
BEHAVIORAL_FEATURES = behavioral_feature_names()

SUITE_TITLES = {
    "vad": "VAD",
    "acoustic": "Acoustic",
    "semantic": "Semantic",
    "behavioral": "Behavioural",
}
PURPOSES = {
    "vad": "Match organizer speech segments so serve-time turns match train-time features.",
    "acoustic": "P(synthetic) from how the caller channel sounds.",
    "semantic": "P(synthetic) from agent-script traps and the caller's reply.",
    "behavioral": "P(synthetic) from who talks when (timing, overlap, recovery).",
}


def _cohens_d(human: list[float], synthetic: list[float]) -> float:
    if not human or not synthetic:
        return 0.0
    delta = mean(synthetic) - mean(human)
    human_std = pstdev(human) if len(human) > 1 else 0.0
    synthetic_std = pstdev(synthetic) if len(synthetic) > 1 else 0.0
    pooled = ((human_std**2 + synthetic_std**2) / 2) ** 0.5
    return delta / pooled if pooled else 0.0


def select_top_features(
    rows: Sequence,
    candidates: Sequence[str],
    k: int = 5,
) -> list[dict]:
    """Rank candidate features by |Cohen's d| on these rows (train only)."""
    ranked: list[dict] = []
    for name in candidates:
        human = [float(getattr(row, name)) for row in rows if row.label == "human"]
        synthetic = [float(getattr(row, name)) for row in rows if row.label == "synthetic"]
        effect = _cohens_d(human, synthetic)
        ranked.append({"feature": name, "effect_size": round(effect, 4)})
    ranked.sort(key=lambda item: abs(item["effect_size"]), reverse=True)
    return ranked[:k]


def fit_and_score(
    train_rows: Sequence,
    val_rows: Sequence,
    candidates: Sequence[str],
    k: int = 5,
) -> dict | None:
    """Train logistic on train top-k features; score val. Selection does not use val."""
    if len(train_rows) < 4 or len(val_rows) < 2:
        return None
    if {row.label for row in train_rows} != {"human", "synthetic"}:
        return None
    if {row.label for row in val_rows} != {"human", "synthetic"}:
        return None

    separators = select_top_features(train_rows, candidates, k=min(k, len(candidates)))
    names = tuple(item["feature"] for item in separators)
    model = train_logistic_regression(list(train_rows), names)
    train_metrics = evaluate_classifier(model, list(train_rows))
    val_metrics = evaluate_classifier(model, list(val_rows))
    return {
        "features_used": list(names),
        "top_separators": separators,
        "train": {
            "accuracy": round(train_metrics.accuracy, 4),
            "f1": round(train_metrics.f1, 4),
            "auc": round(train_metrics.auc, 4),
        },
        "val": {
            "accuracy": round(val_metrics.accuracy, 4),
            "precision": round(val_metrics.precision, 4),
            "recall": round(val_metrics.recall, 4),
            "f1": round(val_metrics.f1, 4),
            "auc": round(val_metrics.auc, 4),
        },
        "num_train": len(train_rows),
        "num_val": len(val_rows),
    }


def _skipped(suite: str, reason: str) -> dict:
    return {
        "id": suite,
        "title": SUITE_TITLES[suite],
        "purpose": PURPOSES[suite],
        "status": "skipped",
        "headline": "skipped",
        "headline_detail": reason,
        "reason": reason,
        "table": None,
        "train": {},
        "val": {},
        "features_used": [],
        "top_separators": [],
    }


def _detection_suite(suite: str, scored: dict | None, empty_reason: str) -> dict:
    if scored is None:
        return _skipped(suite, empty_reason)
    val = scored["val"]
    return {
        "id": suite,
        "title": SUITE_TITLES[suite],
        "purpose": PURPOSES[suite],
        "status": "ok",
        "headline": f"{val['auc']:.3f} val AUC",
        "headline_detail": (
            f"acc {val['accuracy']:.3f} · f1 {val['f1']:.3f} · "
            f"{scored['num_train']} train / {scored['num_val']} val"
        ),
        "reason": "",
        "table": {
            "columns": ["split", "accuracy", "f1", "auc"],
            "rows": [
                ["train", scored["train"]["accuracy"], scored["train"]["f1"], scored["train"]["auc"]],
                ["val", val["accuracy"], val["f1"], val["auc"]],
            ],
        },
        "train": scored["train"],
        "val": val,
        "features_used": scored["features_used"],
        "top_separators": scored["top_separators"],
        "num_train": scored["num_train"],
        "num_val": scored["num_val"],
    }


def _vad_suite(results) -> dict:
    if not results or results[0].num_calls == 0:
        return _skipped("vad", "Dataset not found or empty.")
    best = max(results, key=lambda item: item.mean_overall_iou)
    return {
        "id": "vad",
        "title": SUITE_TITLES["vad"],
        "purpose": PURPOSES["vad"],
        "status": "ok",
        "headline": f"{best.mean_overall_iou:.3f} overall IoU",
        "headline_detail": f"{best.backend} · {best.num_calls} calls",
        "reason": "",
        "table": {
            "columns": [
                "backend",
                "calls",
                "caller_iou",
                "agent_iou",
                "overall",
                "latency_ms",
                "p95_ms",
            ],
            "rows": [
                [
                    item.backend,
                    item.num_calls,
                    round(item.caller.mean_iou, 4),
                    round(item.agent.mean_iou, 4),
                    round(item.mean_overall_iou, 4),
                    round(item.mean_latency_ms, 1),
                    round(item.p95_latency_ms, 1),
                ]
                for item in results
            ],
        },
        "train": {},
        "val": {},
        "features_used": [],
        "top_separators": [],
    }


def _semantic_rows(
    split: str,
    dataset_root: Path | str | None,
    transcripts_dir: Path,
    limit: int | None,
) -> list[SimpleNamespace]:
    rows: list[SimpleNamespace] = []
    try:
        samples = iter_split(split, root=dataset_root, load_audio=False)  # type: ignore[arg-type]
    except DatasetError:
        return rows
    for sample in samples:
        if limit is not None and len(rows) >= limit:
            break
        path = transcripts_dir / f"{sample.anon_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        features = extract_semantic_features(payload.get("turns", []))
        rows.append(SimpleNamespace(label=sample.label, **features))
    return rows


def run_layer_benchmarks(
    *,
    dataset_root: Path | str | None = None,
    split: str = "val",
    limit: int | None = None,
    suites: Iterable[str] = ("vad", "acoustic", "semantic", "behavioral"),
    transcripts_dir: Path | str | None = None,
    show_progress: bool = False,
) -> dict:
    """Run selected layer benchmarks and return a JSON-serializable payload."""
    wanted = tuple(suites)
    started = perf_counter()
    transcript_root = Path(transcripts_dir) if transcripts_dir else Path("transcripts")
    empty = "Dataset not found, or not enough labelled calls."

    train_rows: list = []
    val_rows: list = []
    if {"acoustic", "behavioral"} & set(wanted):
        try:
            train_rows = _collect_features(
                "train", dataset_root, limit=limit, show_progress=show_progress
            )
            val_rows = _collect_features(
                "val", dataset_root, limit=limit, show_progress=show_progress
            )
        except DatasetError:
            train_rows = []
            val_rows = []

    results: list[dict] = []
    for suite in wanted:
        if suite == "vad":
            vad = run_vad_benchmark_all(split=split, dataset_root=dataset_root, limit=limit)
            results.append(_vad_suite(vad))
        elif suite == "acoustic":
            results.append(
                _detection_suite("acoustic", fit_and_score(train_rows, val_rows, ACOUSTIC_FEATURES), empty)
            )
        elif suite == "behavioral":
            results.append(
                _detection_suite(
                    "behavioral",
                    fit_and_score(train_rows, val_rows, BEHAVIORAL_FEATURES),
                    empty,
                )
            )
        elif suite == "semantic":
            if not transcript_root.is_dir():
                results.append(_skipped("semantic", f"No transcript directory at {transcript_root}."))
                continue
            train_sem = _semantic_rows("train", dataset_root, transcript_root, limit)
            val_sem = _semantic_rows("val", dataset_root, transcript_root, limit)
            scored = fit_and_score(train_sem, val_sem, SEMANTIC_FEATURES, k=len(SEMANTIC_FEATURES))
            results.append(
                _detection_suite(
                    "semantic",
                    scored,
                    f"Need labelled transcripts in {transcript_root} for both splits.",
                )
            )
        else:
            raise ValueError(f"Unknown suite: {suite}")

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "duration_s": round(perf_counter() - started, 2),
        "split": split,
        "limit": limit,
        "suites": results,
    }


def format_benchmark_text(payload: dict) -> str:
    lines = ["Layer benchmarks (dataset, not unit tests)", ""]
    for suite in payload["suites"]:
        line = f"{suite['title']:<12} {suite['status']:<8} {suite['headline']}"
        if suite.get("headline_detail"):
            line += f"  ({suite['headline_detail']})"
        lines.append(line)
        if suite["status"] == "ok" and suite.get("features_used"):
            lines.append(f"{'':12} features: {', '.join(suite['features_used'])}")
    return "\n".join(lines)
