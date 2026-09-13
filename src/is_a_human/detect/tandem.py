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
from is_a_human.analysis.semantic import SEMANTIC_HEAD_FEATURES
from is_a_human.eval.layer_benchmark import select_top_features
from is_a_human.turns.backends import DEFAULT_VAD_BACKEND, VadBackendName

DEFAULT_MODEL_PATH = Path("models/tandem.json")
STACK_MARGIN = 0.01
FUSION_FEATURES = ("p_acoustic", "p_behavioral")
FUSION3_FEATURES = ("p_acoustic", "p_behavioral", "p_semantic")
# When heads disagree and the mixer is near 0.5, use the sharper head.
DISAGREE_UNSURE = 0.10
# Fast path answers when |p - 0.5| >= GATE_MARGIN and the two fast heads agree.
# Otherwise the caller is transcribed and the semantic head joins in. Both
# margins were read off train OOF (see reports/robustness/): the unsure gate
# alone lifts 95.7 -> 97.6; adding the disagreement trigger costs ~3 points of
# extra Whisper calls and lifts a flipped-behavioural-head scenario from 50 to
# 68 -- a confidently wrong head never looks unsure, so without the trigger the
# semantic head would never be consulted exactly when it is needed.
GATE_MARGIN = 0.30
DISAGREE_MARGIN = 0.15
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
    semantic: TrainedLogistic | None = None
    fusion3: TrainedLogistic | None = None      # fit for the report; not used to decide
    gate_margin: float = GATE_MARGIN
    disagree_margin: float = DISAGREE_MARGIN

    # ---- head availability --------------------------------------------------
    @staticmethod
    def head_availability(features: CallFeatures) -> dict[str, bool]:
        """Which fast heads can be trusted for this call.

        The behavioural head is made of turn boundaries and the acoustic head
        of frames inside caller segments, so when the VAD finds no caller
        speech both are reading nothing. A call under a second of caller
        speech gives the acoustic statistics almost no frames either."""
        duration_s = float(getattr(features, "duration_s", 0.0))
        caller_segments = float(getattr(features, "caller_segment_count", 0.0))
        caller_ratio = float(getattr(features, "caller_talk_ratio", 0.0))
        caller_speech_s = caller_ratio * duration_s
        # Only a real call (positive duration) with no caller speech at all counts
        # as a dead VAD; synthetic feature rows in tests carry no duration.
        vad_dead = duration_s > 0 and caller_segments <= 0 and caller_ratio <= 0
        return {
            "acoustic": not vad_dead and not (duration_s > 0 and caller_speech_s < 1.0),
            "behavioral": not vad_dead,
        }

    def neutral(self, head: str) -> float:
        """The probability at which a head adds nothing to the fusion: its
        training mean, which the fusion's scaler maps to zero."""
        idx = {"acoustic": 0, "behavioral": 1}[head]
        return float(self.fusion.mean[idx]) if self.fusion is not None else 0.5

    # ---- three-head path ---------------------------------------------------
    def has_semantic(self) -> bool:
        return self.semantic is not None

    def heads_disagree(self, views: dict[str, float]) -> bool:
        """Both fast heads available, on opposite sides of 0.5, each by a margin."""
        if views.get("acoustic_ok", 1.0) < 1.0 or views.get("behavioral_ok", 1.0) < 1.0:
            return False
        a, b = views["acoustic"] - 0.5, views["behavioral"] - 0.5
        return a * b < 0 and abs(a) > self.disagree_margin and abs(b) > self.disagree_margin

    def needs_semantic(
        self,
        p_fast: float,
        availability: dict[str, bool] | None = None,
        views: dict[str, float] | None = None,
    ) -> bool:
        """Consult the semantic head when the fast path is unsure, when its two
        heads contradict each other, or when one of them was masked."""
        if not self.has_semantic():
            return False
        if availability is not None and not all(availability.values()):
            return True
        if views is not None and self.heads_disagree(views):
            return True
        return abs(p_fast - 0.5) < self.gate_margin

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    def predict_full(self, features: CallFeatures) -> tuple[float, dict[str, float]]:
        """Fast fusion and semantic head combined as the mean of their log-odds.

        Parameter-free on purpose: a fitted three-way combiner scored the same
        on clean train OOF and worse whenever a head was masked or corrupted
        (reports/robustness/). Falls back to the fast path when the row has no
        transcript (semantic_available == 0)."""
        p_fast, views = self.predict(features)
        if not self.has_semantic() or float(getattr(features, "semantic_available", 0.0)) < 1.0:
            return p_fast, views
        views = dict(views)
        views["semantic"] = self.semantic.predict_one(features)
        z = (self._logit(p_fast) + self._logit(views["semantic"])) / 2.0
        return float(1.0 / (1.0 + np.exp(-z))), views

    def predict(self, features: CallFeatures) -> tuple[float, dict[str, float]]:
        avail = self.head_availability(features)
        views = {
            "acoustic": self.acoustic.predict_one(features) if avail["acoustic"] else self.neutral("acoustic"),
            "behavioral": self.behavioral.predict_one(features) if avail["behavioral"] else self.neutral("behavioral"),
            "acoustic_ok": 1.0 if avail["acoustic"] else 0.0,
            "behavioral_ok": 1.0 if avail["behavioral"] else 0.0,
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


def _oof_head_probabilities(
    train_rows: list[CallFeatures],
    heads: dict[str, tuple[str, ...]],
) -> np.ndarray:
    """Out-of-fold P(synthetic) per head, shape (rows, heads). Each head is refit
    on four fifths and scores the fifth it did not see."""
    labels = [row.label for row in train_rows]
    names = list(heads)
    oof = np.full((len(train_rows), len(names)), 0.5)
    for train_idx, val_idx in _stratified_folds(labels, n_splits=5):
        fold_train = [train_rows[i] for i in train_idx]
        fold_val = [train_rows[i] for i in val_idx]
        if {row.label for row in fold_train} != {"human", "synthetic"}:
            continue
        for j, name in enumerate(names):
            head = train_logistic_regression(fold_train, heads[name])
            oof[val_idx, j] = head.predict_proba_rows(fold_val)
    return oof


def _fit_stacked_fusion(
    train_rows: list[CallFeatures],
    acoustic_names: tuple[str, ...],
    behavioral_names: tuple[str, ...],
    semantic_names: tuple[str, ...] | None = None,
) -> TrainedLogistic:
    """Two-head fusion, or three-head when semantic_names is given."""
    heads = {"p_acoustic": acoustic_names, "p_behavioral": behavioral_names}
    if semantic_names:
        heads["p_semantic"] = semantic_names
    oof = _oof_head_probabilities(train_rows, heads)
    fusion_rows = [
        SimpleNamespace(label=row.label, **{k: float(oof[i, j]) for j, k in enumerate(heads)})
        for i, row in enumerate(train_rows)
    ]
    return train_logistic_regression(fusion_rows, tuple(heads))


def _gate_report(
    train_rows: list[CallFeatures],
    acoustic_names: tuple[str, ...],
    behavioral_names: tuple[str, ...],
    margin: float,
) -> dict:
    """What the gate does on train OOF: how many calls it sends to Whisper and
    how many of the fast path's misses it catches."""
    oof = _oof_head_probabilities(train_rows, {"p_acoustic": acoustic_names, "p_behavioral": behavioral_names})
    fusion_rows = [SimpleNamespace(label=r.label, p_acoustic=float(oof[i, 0]), p_behavioral=float(oof[i, 1]))
                   for i, r in enumerate(train_rows)]
    fusion = train_logistic_regression(fusion_rows, FUSION_FEATURES)
    p = fusion.predict_proba_rows(fusion_rows)
    y = np.array([1.0 if r.label == "synthetic" else 0.0 for r in train_rows])
    miss = (p >= 0.5) != (y == 1)
    gated = np.abs(p - 0.5) < margin
    return {
        "margin": margin,
        "train_oof_gated_fraction": round(float(gated.mean()), 4),
        "train_oof_fast_misses": int(miss.sum()),
        "train_oof_misses_inside_gate": int((miss & gated).sum()),
    }


def train_tandem_from_rows(
    train_rows: list[CallFeatures],
    val_rows: list[CallFeatures],
    *,
    acoustic_features: tuple[str, ...] = ACOUSTIC_HEAD_FEATURES,
    behavioral_features: tuple[str, ...] = BEHAVIORAL_HEAD_FEATURES,
    semantic_features: tuple[str, ...] = SEMANTIC_HEAD_FEATURES,
    vad_backend: VadBackendName = DEFAULT_VAD_BACKEND,
    gate_margin: float = GATE_MARGIN,
    disagree_margin: float = DISAGREE_MARGIN,
) -> TandemModel:
    """Fit heads + both fusion styles. Ship stacked if within STACK_MARGIN of concat val AUC.

    When the training rows carry transcripts (semantic_available == 1), a third
    head and a three-head fusion are fit as well, and the model gates between
    the two at serve time."""
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

    # ---- semantic head + three-head fusion, only when transcripts exist -----
    sem_train = [r for r in train_rows if float(getattr(r, "semantic_available", 0.0)) >= 1.0]
    sem_val = [r for r in val_rows if float(getattr(r, "semantic_available", 0.0)) >= 1.0]
    if semantic_features and len(sem_train) >= 40 and {r.label for r in sem_train} == {"human", "synthetic"}:
        semantic_names = tuple(semantic_features)
        model.semantic = train_logistic_regression(sem_train, semantic_names)
        model.fusion3 = _fit_stacked_fusion(sem_train, acoustic_names, behavioral_names, semantic_names)
        model.gate_margin = gate_margin
        model.disagree_margin = disagree_margin
        model.metrics["semantic"] = {
            "val": _metrics_dict(model.semantic, sem_val) if sem_val else {},
            "features_used": list(semantic_names),
            "train_rows_with_transcript": len(sem_train),
        }
        model.metrics["gate"] = _gate_report(train_rows, acoustic_names, behavioral_names, gate_margin)
        if sem_val:
            y_true = np.array([1.0 if r.label == "synthetic" else 0.0 for r in sem_val])
            y_full = np.array([model.predict_full(r)[0] for r in sem_val])
            def _gated(r):
                p, v = model.predict(r)
                return model.predict_full(r)[0] if model.needs_semantic(p, model.head_availability(r), v) else p
            y_gated = np.array([_gated(r) for r in sem_val])
            model.metrics["stacked3"] = {"val": {
                "always": {k: round(float(v), 4) for k, v in
                           _compute_metrics(y_true, (y_full >= 0.5).astype(int), y_full).__dict__.items()
                           if k in ("accuracy", "f1", "auc")},
                "gated": {k: round(float(v), 4) for k, v in
                          _compute_metrics(y_true, (y_gated >= 0.5).astype(int), y_gated).__dict__.items()
                          if k in ("accuracy", "f1", "auc")},
                "gated_fraction": round(float(np.mean([
                    model.needs_semantic(model.predict(r)[0], model.head_availability(r), model.predict(r)[1])
                    for r in sem_val])), 4),
            }}
            model.metrics["fusion3_weights"] = {
                k: round(float(w), 4) for k, w in zip(FUSION3_FEATURES, model.fusion3.weights)
            } | {"bias": round(float(model.fusion3.bias), 4)}
    return model


def train_tandem(
    dataset_root: Path | str | None = None,
    *,
    limit: int | None = None,
    show_progress: bool = False,
    vad_backend: VadBackendName = DEFAULT_VAD_BACKEND,
    acoustic_features: tuple[str, ...] = ACOUSTIC_HEAD_FEATURES,
    behavioral_features: tuple[str, ...] = BEHAVIORAL_HEAD_FEATURES,
    semantic_features: tuple[str, ...] = SEMANTIC_HEAD_FEATURES,
    transcripts_dir: Path | str | None = None,
    gate_margin: float = GATE_MARGIN,
) -> TandemModel:
    train_rows = _collect_features(
        "train",
        dataset_root,
        limit=limit,
        show_progress=show_progress,
        heavy=False,
        vad_backend=vad_backend,
        transcripts_dir=transcripts_dir,
    )
    val_rows = _collect_features(
        "val",
        dataset_root,
        limit=limit,
        show_progress=show_progress,
        heavy=False,
        vad_backend=vad_backend,
        transcripts_dir=transcripts_dir,
    )
    return train_tandem_from_rows(
        train_rows,
        val_rows,
        acoustic_features=acoustic_features,
        behavioral_features=behavioral_features,
        semantic_features=semantic_features,
        vad_backend=vad_backend,
        gate_margin=gate_margin,
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
        "semantic": model.semantic.to_dict() if model.semantic else None,
        "fusion3": model.fusion3.to_dict() if model.fusion3 else None,
        "gate_margin": model.gate_margin,
        "disagree_margin": model.disagree_margin,
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
        semantic=TrainedLogistic.from_dict(payload["semantic"]) if payload.get("semantic") else None,
        fusion3=TrainedLogistic.from_dict(payload["fusion3"]) if payload.get("fusion3") else None,
        gate_margin=float(payload.get("gate_margin", GATE_MARGIN)),
        disagree_margin=float(payload.get("disagree_margin", DISAGREE_MARGIN)),
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
