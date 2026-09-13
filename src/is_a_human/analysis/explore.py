"""Exploratory analysis: compare conversational features by label."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

import numpy as np

from is_a_human.analysis.classifier import (
    ClassifierMetrics,
    evaluate_classifier,
    train_logistic_regression,
)
from is_a_human.analysis.features import CallFeatures, extract_call_features
from is_a_human.dataset.loader import DatasetError, count_split, iter_split

Split = Literal["train", "val"]


class _ExploreProgress:
    """stderr progress lines for long-running explore jobs."""

    def __init__(self, phase: str, total: int | None, *, enabled: bool = True) -> None:
        self.phase = phase
        self.total = total
        self.enabled = enabled
        self.count = 0
        self._start = time.monotonic()
        self._use_carriage = enabled and sys.stderr.isatty()

    def _format_eta(self, elapsed: float) -> str:
        if self.total is None or self.count <= 0:
            return "eta ?"
        remaining = elapsed / self.count * (self.total - self.count)
        return f"eta {remaining:.0f}s"

    def update(self, detail: str = "") -> None:
        if not self.enabled:
            return

        self.count += 1
        elapsed = time.monotonic() - self._start
        if self.total is not None:
            pct = 100.0 * self.count / self.total
            head = f"[explore] {self.phase}: {self.count}/{self.total} ({pct:.0f}%)"
        else:
            head = f"[explore] {self.phase}: {self.count}"

        tail = f"{elapsed:.0f}s elapsed, {self._format_eta(elapsed)}"
        msg = f"{head} — {detail} — {tail}" if detail else f"{head} — {tail}"

        if self._use_carriage:
            sys.stderr.write(f"\r{msg:<120}")
        elif self.count == 1 or self.count % 5 == 0 or self.count == self.total:
            sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    def message(self, text: str) -> None:
        if not self.enabled:
            return
        if self._use_carriage and self.count > 0:
            sys.stderr.write("\n")
        sys.stderr.write(f"[explore] {text}\n")
        sys.stderr.flush()

    def done(self) -> None:
        if self.enabled and self._use_carriage and self.count > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()


@dataclass(frozen=True)
class FeatureComparison:
    feature: str
    human_mean: float
    human_std: float
    synthetic_mean: float
    synthetic_std: float
    delta: float
    effect_size: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float


@dataclass(frozen=True)
class ExploreSummary:
    split: Split
    num_calls: int
    num_human: int
    num_synthetic: int
    comparisons: tuple[FeatureComparison, ...]
    top_separators: tuple[FeatureComparison, ...]


@dataclass(frozen=True)
class MultiRunSummary:
    train: ExploreSummary
    val: ExploreSummary
    stable_top_features: tuple[str, ...]
    split_rank_correlation: float
    classifier_train: ClassifierMetrics
    classifier_val: ClassifierMetrics
    bootstrap_runs: int


def _group_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _compare_feature(
    feature: str,
    human_values: list[float],
    synthetic_values: list[float],
    *,
    bootstrap_effects: list[float] | None = None,
) -> FeatureComparison:
    human_mean, human_std = _group_stats(human_values)
    synthetic_mean, synthetic_std = _group_stats(synthetic_values)
    delta = synthetic_mean - human_mean

    pooled_std = 0.0
    if human_values and synthetic_values:
        human_var = human_std**2
        synthetic_var = synthetic_std**2
        pooled_std = ((human_var + synthetic_var) / 2) ** 0.5

    effect_size = delta / pooled_std if pooled_std > 0 else 0.0

    if bootstrap_effects:
        ci_low = float(np.quantile(bootstrap_effects, 0.025))
        ci_high = float(np.quantile(bootstrap_effects, 0.975))
    else:
        ci_low = effect_size
        ci_high = effect_size

    return FeatureComparison(
        feature=feature,
        human_mean=human_mean,
        human_std=human_std,
        synthetic_mean=synthetic_mean,
        synthetic_std=synthetic_std,
        delta=delta,
        effect_size=effect_size,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
    )


def _collect_features(
    split: Split,
    dataset_root: Path | str | None,
    *,
    limit: int | None = None,
    show_progress: bool = False,
) -> list[CallFeatures]:
    try:
        total = count_split(split, dataset_root)
    except DatasetError:
        total = None
    if limit is not None and total is not None:
        total = min(total, limit)

    progress = _ExploreProgress(f"features/{split}", total, enabled=show_progress)
    features: list[CallFeatures] = []
    for sample in iter_split(split, root=dataset_root, load_audio=True):
        if limit is not None and len(features) >= limit:
            break
        features.append(
            extract_call_features(
                anon_id=sample.anon_id,
                label=sample.label,
                split=sample.split,
                ch0_caller=sample.ch0_caller,
                ch1_agent=sample.ch1_agent,
                sample_rate=sample.sample_rate,
            )
        )
        progress.update(f"{sample.anon_id} ({sample.label})")
    progress.done()
    return features


def _bootstrap_effects(
    rows: list[CallFeatures],
    feature: str,
    *,
    runs: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    labels = np.array([row.label for row in rows])
    effects: list[float] = []

    for _ in range(runs):
        indices = rng.integers(0, len(rows), len(rows))
        sample = [rows[index] for index in indices]
        human_values = [float(getattr(row, feature)) for row in sample if row.label == "human"]
        synthetic_values = [float(getattr(row, feature)) for row in sample if row.label == "synthetic"]
        if not human_values or not synthetic_values:
            continue
        comparison = _compare_feature(feature, human_values, synthetic_values)
        effects.append(comparison.effect_size)

    return effects


def _summarize_split(
    split: Split,
    rows: list[CallFeatures],
    *,
    bootstrap_runs: int = 0,
    seed: int = 42,
    show_progress: bool = False,
) -> ExploreSummary:
    human_rows = [row for row in rows if row.label == "human"]
    synthetic_rows = [row for row in rows if row.label == "synthetic"]

    field_names = CallFeatures.numeric_field_names()
    progress = _ExploreProgress(
        f"bootstrap/{split}",
        len(field_names) if bootstrap_runs > 0 else None,
        enabled=show_progress and bootstrap_runs > 0,
    )

    comparisons: list[FeatureComparison] = []
    for field in field_names:
        human_values = [float(getattr(row, field)) for row in human_rows]
        synthetic_values = [float(getattr(row, field)) for row in synthetic_rows]
        bootstrap = (
            _bootstrap_effects(rows, field, runs=bootstrap_runs, seed=seed)
            if bootstrap_runs > 0
            else None
        )
        comparisons.append(
            _compare_feature(field, human_values, synthetic_values, bootstrap_effects=bootstrap)
        )
        if bootstrap_runs > 0:
            progress.update(field)

    progress.done()
    ranked = sorted(comparisons, key=lambda item: abs(item.effect_size), reverse=True)
    return ExploreSummary(
        split=split,
        num_calls=len(rows),
        num_human=len(human_rows),
        num_synthetic=len(synthetic_rows),
        comparisons=tuple(comparisons),
        top_separators=tuple(ranked[:8]),
    )


def _rank_map(summary: ExploreSummary) -> dict[str, int]:
    ranked = sorted(summary.comparisons, key=lambda item: abs(item.effect_size), reverse=True)
    return {item.feature: index for index, item in enumerate(ranked)}


def _rank_correlation(train: ExploreSummary, val: ExploreSummary) -> float:
    shared = set(_rank_map(train)) & set(_rank_map(val))
    if len(shared) < 2:
        return 0.0
    train_ranks = np.array([_rank_map(train)[feature] for feature in shared], dtype=float)
    val_ranks = np.array([_rank_map(val)[feature] for feature in shared], dtype=float)
    if np.std(train_ranks) == 0 or np.std(val_ranks) == 0:
        return 0.0
    return float(np.corrcoef(train_ranks, val_ranks)[0, 1])


def _stable_features(train: ExploreSummary, val: ExploreSummary, top_k: int = 5) -> tuple[str, ...]:
    train_top = {item.feature for item in train.top_separators[:top_k]}
    val_top = {item.feature for item in val.top_separators[:top_k]}
    shared = train_top & val_top
    ranked = sorted(
        shared,
        key=lambda name: abs(next(item.effect_size for item in train.comparisons if item.feature == name)),
        reverse=True,
    )
    return tuple(ranked)


def run_exploratory_analysis(
    split: Split = "val",
    dataset_root: Path | str | None = None,
    *,
    limit: int | None = None,
    bootstrap_runs: int = 0,
    show_progress: bool = False,
) -> ExploreSummary:
    """Extract features and compare human vs synthetic on a split."""
    try:
        rows = _collect_features(split, dataset_root, limit=limit, show_progress=show_progress)
    except DatasetError:
        return ExploreSummary(
            split=split,
            num_calls=0,
            num_human=0,
            num_synthetic=0,
            comparisons=tuple(),
            top_separators=tuple(),
        )

    return _summarize_split(
        split, rows, bootstrap_runs=bootstrap_runs, show_progress=show_progress
    )


def run_multi_analysis(
    dataset_root: Path | str | None = None,
    *,
    limit: int | None = None,
    bootstrap_runs: int = 200,
    top_features: int = 5,
    show_progress: bool = False,
) -> MultiRunSummary:
    """Run train + val analysis, stability check, and logistic baseline."""
    progress = _ExploreProgress("multi", None, enabled=show_progress)

    train_rows = _collect_features(
        "train", dataset_root, limit=limit, show_progress=show_progress
    )
    val_rows = _collect_features("val", dataset_root, limit=limit, show_progress=show_progress)

    train_summary = _summarize_split(
        "train", train_rows, bootstrap_runs=bootstrap_runs, show_progress=show_progress
    )
    val_summary = _summarize_split(
        "val", val_rows, bootstrap_runs=bootstrap_runs, show_progress=show_progress
    )

    progress.message("training logistic baseline...")
    stable = _stable_features(train_summary, val_summary, top_k=top_features)
    if not stable:
        stable = tuple(item.feature for item in train_summary.top_separators[:top_features])

    model = train_logistic_regression(train_rows, stable)
    train_metrics = evaluate_classifier(model, train_rows)
    val_metrics = evaluate_classifier(model, val_rows)
    progress.message("done")
    progress.done()

    return MultiRunSummary(
        train=train_summary,
        val=val_summary,
        stable_top_features=stable,
        split_rank_correlation=_rank_correlation(train_summary, val_summary),
        classifier_train=train_metrics,
        classifier_val=val_metrics,
        bootstrap_runs=bootstrap_runs,
    )


def format_explore_report(summary: ExploreSummary) -> str:
    """Render a human-readable exploratory analysis report."""
    lines = [
        f"split={summary.split} calls={summary.num_calls} "
        f"human={summary.num_human} synthetic={summary.num_synthetic}",
        "",
        "Top separators (by |effect size|):",
    ]

    for item in summary.top_separators:
        direction = "synthetic > human" if item.delta > 0 else "human > synthetic"
        lines.append(
            f"  {item.feature}: "
            f"human={item.human_mean:.3f}±{item.human_std:.3f} "
            f"synthetic={item.synthetic_mean:.3f}±{item.synthetic_std:.3f} "
            f"d={item.effect_size:+.2f} [{item.bootstrap_ci_low:+.2f}, {item.bootstrap_ci_high:+.2f}] "
            f"({direction})"
        )

    return "\n".join(lines)


def format_multi_report(summary: MultiRunSummary) -> str:
    lines = [
        "=== Multi-run exploratory analysis ===",
        "",
        format_explore_report(summary.train),
        "",
        format_explore_report(summary.val),
        "",
        f"split_rank_correlation={summary.split_rank_correlation:.3f}",
        f"stable_top_features={', '.join(summary.stable_top_features)}",
        f"bootstrap_runs={summary.bootstrap_runs}",
        "",
        "Logistic baseline (train on train, evaluate on val):",
        f"  train accuracy={summary.classifier_train.accuracy:.3f} "
        f"f1={summary.classifier_train.f1:.3f} auc={summary.classifier_train.auc:.3f}",
        f"  val   accuracy={summary.classifier_val.accuracy:.3f} "
        f"f1={summary.classifier_val.f1:.3f} auc={summary.classifier_val.auc:.3f}",
    ]
    return "\n".join(lines)
