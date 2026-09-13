"""Tandem detector: VAD-backed acoustic + behavioural heads, then fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from is_a_human.analysis.classifier import (
    TrainedLogistic,
    _compute_metrics,
    evaluate_classifier,
    train_logistic_regression,
)
from is_a_human.analysis.explore import _collect_features
from is_a_human.analysis.features import (
    ACOUSTIC_HEAD_FEATURES,
    BEHAVIORAL_HEAD_FEATURES,
    CallFeatures,
)
from is_a_human.eval.layer_benchmark import select_top_features
from is_a_human.turns.backends import DEFAULT_VAD_BACKEND, VadBackendName

DEFAULT_MODEL_PATH = Path("models/tandem.json")
STACK_MARGIN = 0.01
FUSION_FEATURES = ("p_acoustic", "p_behavioral")
# When heads disagree and the mixer is near 0.5, use the sharper head.
DISAGREE_UNSURE = 0.10
# Quiet-patient gate: wait and tidy energy. Talkative subtype (many caller
# turns) is overridden to human. Cuts frozen on train-box occupancy; k=24 sits
# above talkative train-box synths (21–22 hybrid turns). Set talkative_turn_min
# to 0 to disable.
QUIET_WAIT_MIN_S = 1.4
QUIET_WAIT_MAX_S = 2.0
QUIET_RMS_CV_MAX = 0.45
TALKATIVE_TURN_MIN = 24


def resolve_disagreement(
    p_fused: float,
    p_acoustic: float,
    p_behavioral: float,
    unsure: float = DISAGREE_UNSURE,
) -> float:
    """Keep the mixer unless it is unsure under a head disagreement."""
    acoustic_synth = p_acoustic >= 0.5
    behavioral_synth = p_behavioral >= 0.5
    if acoustic_synth == behavioral_synth:
        return p_fused
    if abs(p_fused - 0.5) >= unsure:
        return p_fused
    if abs(p_acoustic - 0.5) >= abs(p_behavioral - 0.5):
        return p_acoustic
    return p_behavioral


def resolve_talkative_quiet(
    p_fused: float,
    wait_s: float,
    rms_cv: float,
    caller_segment_count: float,
    *,
    wait_min_s: float = QUIET_WAIT_MIN_S,
    wait_max_s: float = QUIET_WAIT_MAX_S,
    rms_cv_max: float = QUIET_RMS_CV_MAX,
    talkative_turn_min: int = TALKATIVE_TURN_MIN,
) -> float:
    """Override quiet-patient calls with many caller turns to human.

    Gate is class-agnostic (wait + tidy rms_cv). The expert only fires for the
    talkative subtype. Sparse-quiet calls keep the fused score. Disable with
    talkative_turn_min <= 0.
    """
    if talkative_turn_min <= 0 or p_fused < 0.5:
        return p_fused
    if not (wait_min_s <= wait_s <= wait_max_s):
        return p_fused
    if rms_cv >= rms_cv_max:
        return p_fused
    if caller_segment_count < talkative_turn_min:
        return p_fused
    flipped = 1.0 - p_fused
    return flipped if flipped < 0.5 else 0.499


def _quiet_patient_payload(model: "TandemModel") -> dict:
    return {
        "wait_min_s": model.quiet_wait_min_s,
        "wait_max_s": model.quiet_wait_max_s,
        "rms_cv_max": model.quiet_rms_cv_max,
        "talkative_turn_min": model.talkative_turn_min,
    }


@dataclass
class TandemModel:
    fusion_type: str
    vad_backend: str
    acoustic: TrainedLogistic
    behavioral: TrainedLogistic
    fusion: TrainedLogistic | None
    concatenated: TrainedLogistic | None
    metrics: dict
    disagree_unsure: float = DISAGREE_UNSURE
    quiet_wait_min_s: float = QUIET_WAIT_MIN_S
    quiet_wait_max_s: float = QUIET_WAIT_MAX_S
    quiet_rms_cv_max: float = QUIET_RMS_CV_MAX
    talkative_turn_min: int = TALKATIVE_TURN_MIN

    def predict(self, features: CallFeatures) -> tuple[float, dict[str, float]]:
        views = {
            "acoustic": self.acoustic.predict_one(features),
            "behavioral": self.behavioral.predict_one(features),
        }
        if self.fusion_type == "stacked" and self.fusion is not None:
            stacked = SimpleNamespace(
                label=features.label,
                p_acoustic=views["acoustic"],
                p_behavioral=views["behavioral"],
            )
            fused = self.fusion.predict_one(stacked)
            fused = resolve_disagreement(
                fused, views["acoustic"], views["behavioral"], self.disagree_unsure
            )
            return self._apply_talkative_quiet(fused, features), views
        if self.concatenated is None:
            raise RuntimeError("concatenated model missing")
        fused = self.concatenated.predict_one(features)
        return self._apply_talkative_quiet(fused, features), views

    def _apply_talkative_quiet(self, fused: float, features: CallFeatures) -> float:
        return resolve_talkative_quiet(
            fused,
            float(features.caller_response_latency_pos_median_s),
            float(features.caller_rms_cv),
            float(features.caller_segment_count),
            wait_min_s=self.quiet_wait_min_s,
            wait_max_s=self.quiet_wait_max_s,
            rms_cv_max=self.quiet_rms_cv_max,
            talkative_turn_min=self.talkative_turn_min,
        )


def _metrics_dict(model: TrainedLogistic, rows: list) -> dict:
    scored = evaluate_classifier(model, rows)
    return {
        "accuracy": round(scored.accuracy, 4),
        "f1": round(scored.f1, 4),
        "auc": round(scored.auc, 4),
    }


def _metrics_from_predict(model: TandemModel, rows: list[CallFeatures]) -> dict:
    y_true = np.array([1.0 if row.label == "synthetic" else 0.0 for row in rows])
    y_prob = np.array([model.predict(row)[0] for row in rows])
    y_pred = (y_prob >= 0.5).astype(int)
    scored = _compute_metrics(y_true, y_pred, y_prob)
    return {
        "accuracy": round(scored.accuracy, 4),
        "f1": round(scored.f1, 4),
        "auc": round(float(scored.auc), 4),
    }


def _stratified_folds(labels: list[str], n_splits: int, seed: int = 0):
    by_label: dict[str, list[int]] = {"human": [], "synthetic": []}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    rng = np.random.default_rng(seed)
    for label in by_label:
        rng.shuffle(by_label[label])
    n_splits = max(2, min(n_splits, min(len(idxs) for idxs in by_label.values() if idxs) or 2))
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for idxs in by_label.values():
        for position, index in enumerate(idxs):
            folds[position % n_splits].append(index)
    for holdout in range(n_splits):
        val_idx = folds[holdout]
        train_idx = [index for fold, members in enumerate(folds) if fold != holdout for index in members]
        if train_idx and val_idx:
            yield train_idx, val_idx


def _fit_stacked_fusion(
    train_rows: list[CallFeatures],
    acoustic_names: tuple[str, ...],
    behavioral_names: tuple[str, ...],
) -> TrainedLogistic:
    labels = [row.label for row in train_rows]
    oof = np.full((len(train_rows), 2), 0.5)
    for train_idx, val_idx in _stratified_folds(labels, n_splits=5):
        fold_train = [train_rows[i] for i in train_idx]
        fold_val = [train_rows[i] for i in val_idx]
        if {row.label for row in fold_train} != {"human", "synthetic"}:
            continue
        acoustic = train_logistic_regression(fold_train, acoustic_names)
        behavioral = train_logistic_regression(fold_train, behavioral_names)
        oof[val_idx, 0] = acoustic.predict_proba_rows(fold_val)
        oof[val_idx, 1] = behavioral.predict_proba_rows(fold_val)
    fusion_rows = [
        SimpleNamespace(label=row.label, p_acoustic=float(oof[i, 0]), p_behavioral=float(oof[i, 1]))
        for i, row in enumerate(train_rows)
    ]
    return train_logistic_regression(fusion_rows, FUSION_FEATURES)


def train_tandem_from_rows(
    train_rows: list[CallFeatures],
    val_rows: list[CallFeatures],
    *,
    acoustic_features: tuple[str, ...] = ACOUSTIC_HEAD_FEATURES,
    behavioral_features: tuple[str, ...] = BEHAVIORAL_HEAD_FEATURES,
    vad_backend: VadBackendName = DEFAULT_VAD_BACKEND,
) -> TandemModel:
    """Fit heads + both fusion styles. Ship stacked if within STACK_MARGIN of concat val AUC."""
    acoustic_names = tuple(acoustic_features)
    behavioral_names = tuple(behavioral_features)
    if not acoustic_names or not behavioral_names:
        raise ValueError("Need labelled human and synthetic rows in train and val.")
    if {row.label for row in train_rows} != {"human", "synthetic"}:
        raise ValueError("Need labelled human and synthetic rows in train and val.")
    if {row.label for row in val_rows} != {"human", "synthetic"}:
        raise ValueError("Need labelled human and synthetic rows in train and val.")

    acoustic = train_logistic_regression(train_rows, acoustic_names)
    behavioral = train_logistic_regression(train_rows, behavioral_names)
    fusion = _fit_stacked_fusion(train_rows, acoustic_names, behavioral_names)

    concat_names = tuple(dict.fromkeys(acoustic_names + behavioral_names))
    concatenated = train_logistic_regression(train_rows, concat_names)

    stack_rows_val = [
        SimpleNamespace(
            label=row.label,
            p_acoustic=acoustic.predict_one(row),
            p_behavioral=behavioral.predict_one(row),
        )
        for row in val_rows
    ]
    stacked_val = _metrics_dict(fusion, stack_rows_val)
    concat_val = _metrics_dict(concatenated, val_rows)
    fusion_type = "stacked" if stacked_val["auc"] >= concat_val["auc"] - STACK_MARGIN else "concatenated"

    model = TandemModel(
        fusion_type=fusion_type,
        vad_backend=vad_backend,
        acoustic=acoustic,
        behavioral=behavioral,
        fusion=fusion,
        concatenated=concatenated,
        metrics={
            "acoustic": {
                "val": _metrics_dict(acoustic, val_rows),
                "features_used": list(acoustic_names),
                "top_separators": select_top_features(train_rows, acoustic_names, k=len(acoustic_names)),
            },
            "behavioral": {
                "val": _metrics_dict(behavioral, val_rows),
                "features_used": list(behavioral_names),
                "top_separators": select_top_features(train_rows, behavioral_names, k=len(behavioral_names)),
            },
            "stacked": {"val": stacked_val, "top_separators": []},
            "concatenated": {
                "val": concat_val,
                "features_used": list(concat_names),
                "top_separators": select_top_features(train_rows, concat_names, k=len(concat_names)),
            },
            "fusion_weights": {
                "acoustic": round(float(fusion.weights[0]), 4),
                "behavioral": round(float(fusion.weights[1]), 4),
                "bias": round(float(fusion.bias), 4),
            },
        },
        disagree_unsure=DISAGREE_UNSURE,
    )
    if fusion_type == "stacked":
        model.metrics["stacked"]["val"] = _metrics_from_predict(model, val_rows)
        model.metrics["disagree_unsure"] = DISAGREE_UNSURE
        model.metrics["quiet_patient"] = _quiet_patient_payload(model)
    return model


def train_tandem(
    dataset_root: Path | str | None = None,
    *,
    limit: int | None = None,
    show_progress: bool = False,
    vad_backend: VadBackendName = DEFAULT_VAD_BACKEND,
    acoustic_features: tuple[str, ...] = ACOUSTIC_HEAD_FEATURES,
    behavioral_features: tuple[str, ...] = BEHAVIORAL_HEAD_FEATURES,
) -> TandemModel:
    train_rows = _collect_features(
        "train",
        dataset_root,
        limit=limit,
        show_progress=show_progress,
        heavy=False,
        vad_backend=vad_backend,
    )
    val_rows = _collect_features(
        "val",
        dataset_root,
        limit=limit,
        show_progress=show_progress,
        heavy=False,
        vad_backend=vad_backend,
    )
    return train_tandem_from_rows(
        train_rows,
        val_rows,
        acoustic_features=acoustic_features,
        behavioral_features=behavioral_features,
        vad_backend=vad_backend,
    )


def save_tandem(model: TandemModel, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fusion_type": model.fusion_type,
        "vad_backend": model.vad_backend,
        "disagree_unsure": model.disagree_unsure,
        "quiet_patient": _quiet_patient_payload(model),
        "acoustic": model.acoustic.to_dict(),
        "behavioral": model.behavioral.to_dict(),
        "fusion": model.fusion.to_dict() if model.fusion else None,
        "concatenated": model.concatenated.to_dict() if model.concatenated else None,
        "metrics": model.metrics,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def load_tandem(path: Path | str = DEFAULT_MODEL_PATH) -> TandemModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fusion_payload = payload.get("fusion")
    concat_payload = payload.get("concatenated")
    quiet = payload.get("quiet_patient") or {}
    return TandemModel(
        fusion_type=payload["fusion_type"],
        vad_backend=payload.get("vad_backend", DEFAULT_VAD_BACKEND),
        acoustic=TrainedLogistic.from_dict(payload["acoustic"]),
        behavioral=TrainedLogistic.from_dict(payload["behavioral"]),
        fusion=TrainedLogistic.from_dict(fusion_payload) if fusion_payload else None,
        concatenated=TrainedLogistic.from_dict(concat_payload) if concat_payload else None,
        metrics=payload.get("metrics") or {},
        disagree_unsure=float(payload.get("disagree_unsure", DISAGREE_UNSURE)),
        quiet_wait_min_s=float(quiet.get("wait_min_s", QUIET_WAIT_MIN_S)),
        quiet_wait_max_s=float(quiet.get("wait_max_s", QUIET_WAIT_MAX_S)),
        quiet_rms_cv_max=float(quiet.get("rms_cv_max", QUIET_RMS_CV_MAX)),
        talkative_turn_min=int(quiet.get("talkative_turn_min", TALKATIVE_TURN_MIN)),
    )


def tandem_benchmark_suite(model: TandemModel) -> dict:
    stacked = model.metrics.get("stacked", {}).get("val", {})
    concat = model.metrics.get("concatenated", {}).get("val", {})
    acoustic = model.metrics.get("acoustic", {}).get("val", {})
    behavioral = model.metrics.get("behavioral", {}).get("val", {})
    return {
        "id": "tandem",
        "title": "Tandem",
        "purpose": "VAD ledger → acoustic + behavioural scores → fusion P(synthetic).",
        "status": "ok",
        "headline": f"{stacked.get('auc', 0):.3f} stacked val AUC",
        "headline_detail": (
            f"ship {model.fusion_type} · concat {concat.get('auc', 0):.3f} · "
            f"acoustic {acoustic.get('auc', 0):.3f} · behavioural {behavioral.get('auc', 0):.3f}"
        ),
        "reason": "",
        "table": {
            "columns": ["model", "val_accuracy", "val_f1", "val_auc"],
            "rows": [
                ["acoustic", acoustic.get("accuracy"), acoustic.get("f1"), acoustic.get("auc")],
                ["behavioral", behavioral.get("accuracy"), behavioral.get("f1"), behavioral.get("auc")],
                ["stacked", stacked.get("accuracy"), stacked.get("f1"), stacked.get("auc")],
                ["concatenated", concat.get("accuracy"), concat.get("f1"), concat.get("auc")],
            ],
        },
        "train": {},
        "val": stacked if model.fusion_type == "stacked" else concat,
        "features_used": [model.fusion_type, model.vad_backend],
        "top_separators": model.metrics.get("concatenated", {}).get("top_separators") or [],
    }
