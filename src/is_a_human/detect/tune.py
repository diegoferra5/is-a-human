"""Head ablations and before/after snapshots for the shipped tandem model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from is_a_human.analysis.explore import _collect_features
from is_a_human.analysis.features import (
    ACOUSTIC_HEAD_FEATURES,
    ACOUSTIC_HEAD_FEATURES_WITH_RMS_MEAN,
    BEHAVIORAL_HEAD_FEATURES,
)
from is_a_human.detect.tandem import (
    DEFAULT_MODEL_PATH,
    TandemModel,
    save_tandem,
    train_tandem_from_rows,
)
from is_a_human.turns.backends import VadBackendName

BEFORE_MODEL_PATH = Path("models/tandem-before-tune.json")
REPORT_JSON = Path("reports/tune/before-after.json")
REPORT_MD = Path("reports/tune/before-after.md")
# Drop mean loudness if stacked val acc falls by this much or less (~2 val calls).
RMS_MEAN_HOLD = 0.03


def _head_snapshot(model: TandemModel) -> dict:
    metrics = model.metrics
    return {
        "fusion_type": model.fusion_type,
        "vad_backend": model.vad_backend,
        "acoustic": metrics.get("acoustic", {}),
        "behavioral": metrics.get("behavioral", {}),
        "stacked": metrics.get("stacked", {}),
        "concatenated": metrics.get("concatenated", {}),
        "fusion_weights": metrics.get("fusion_weights", {}),
    }


def _load_before_snapshot() -> dict:
    path = Path("reports/tune/before.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = json.loads(BEFORE_MODEL_PATH.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    fusion = payload.get("fusion") or {}
    weights = fusion.get("weights") or [None, None]
    return {
        "id": "before",
        "source": str(BEFORE_MODEL_PATH),
        "fusion_type": payload.get("fusion_type"),
        "vad_backend": payload.get("vad_backend"),
        "acoustic": metrics.get("acoustic", {}),
        "behavioral": metrics.get("behavioral", {}),
        "stacked": metrics.get("stacked", {}),
        "concatenated": metrics.get("concatenated", {}),
        "fusion_weights": {
            "acoustic": weights[0],
            "behavioral": weights[1],
            "bias": fusion.get("bias"),
        },
    }


def _stacked_acc(snapshot: dict) -> float:
    return float(snapshot.get("stacked", {}).get("val", {}).get("accuracy") or 0.0)


def _behavioral_acc(snapshot: dict) -> float:
    return float(snapshot.get("behavioral", {}).get("val", {}).get("accuracy") or 0.0)


def _render_markdown(payload: dict) -> str:
    before = payload["before"]
    chosen = payload["chosen"]
    lines = [
        "# Tandem head tune — before / after",
        "",
        f"Generated {payload['generated_at']}.",
        "",
        "Revert the shipped detector with:",
        "",
        "```",
        "cp models/tandem-before-tune.json models/tandem.json",
        "```",
        "",
        "## Before (shipped)",
        "",
        f"- fusion `{before.get('fusion_type')}` · VAD `{before.get('vad_backend')}`",
        f"- stacked val acc **{before.get('stacked', {}).get('val', {}).get('accuracy')}** "
        f"AUC {before.get('stacked', {}).get('val', {}).get('auc')}",
        f"- acoustic val acc {before.get('acoustic', {}).get('val', {}).get('accuracy')} "
        f"· features `{', '.join(before.get('acoustic', {}).get('features_used') or [])}`",
        f"- behavioural val acc {before.get('behavioral', {}).get('val', {}).get('accuracy')} "
        f"· features `{', '.join(before.get('behavioral', {}).get('features_used') or [])}`",
        "",
        "## Ablations",
        "",
        "| id | VAD | rms_mean | stacked acc | stacked AUC | acoustic acc | behavioural acc |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload["variants"]:
        snap = item["metrics"]
        stacked = snap.get("stacked", {}).get("val", {})
        acoustic = snap.get("acoustic", {}).get("val", {})
        behavioral = snap.get("behavioral", {}).get("val", {})
        lines.append(
            f"| `{item['id']}` | {item['vad_backend']} | {item['include_rms_mean']} | "
            f"{stacked.get('accuracy')} | {stacked.get('auc')} | "
            f"{acoustic.get('accuracy')} | {behavioral.get('accuracy')} |"
        )
    lines.extend(
        [
            "",
            "## After (chosen)",
            "",
            f"- **{payload['chosen_id']}**: {payload['reason']}",
            f"- fusion `{chosen.get('fusion_type')}` · VAD `{chosen.get('vad_backend')}`",
            f"- stacked val acc **{chosen.get('stacked', {}).get('val', {}).get('accuracy')}** "
            f"AUC {chosen.get('stacked', {}).get('val', {}).get('auc')}",
            f"- acoustic `{', '.join(chosen.get('acoustic', {}).get('features_used') or [])}`",
            f"- behavioural `{', '.join(chosen.get('behavioral', {}).get('features_used') or [])}`",
            f"- mixer weights acoustic={chosen.get('fusion_weights', {}).get('acoustic')} "
            f"behavioural={chosen.get('fusion_weights', {}).get('behavioral')}",
            "",
        ]
    )
    return "\n".join(lines)


def run_head_tune(
    *,
    dataset_root: Path | str | None = None,
    limit: int | None = None,
    show_progress: bool = True,
    output: Path = DEFAULT_MODEL_PATH,
) -> TandemModel:
    """Compare rms_mean on/off and hybrid vs Silero; ship the winner and write reports."""
    before = _load_before_snapshot()
    backends: tuple[VadBackendName, ...] = ("hybrid", "silero")
    rows_by_backend: dict[str, tuple[list, list]] = {}
    for backend in backends:
        train_rows = _collect_features(
            "train",
            dataset_root,
            limit=limit,
            show_progress=show_progress,
            heavy=False,
            vad_backend=backend,
        )
        val_rows = _collect_features(
            "val",
            dataset_root,
            limit=limit,
            show_progress=show_progress,
            heavy=False,
            vad_backend=backend,
        )
        rows_by_backend[backend] = (train_rows, val_rows)

    variants: list[dict] = []
    models: dict[str, TandemModel] = {}
    for backend, (train_rows, val_rows) in rows_by_backend.items():
        for include_rms, acoustic_features in (
            (False, ACOUSTIC_HEAD_FEATURES),
            (True, ACOUSTIC_HEAD_FEATURES_WITH_RMS_MEAN),
        ):
            variant_id = f"{backend}-{'with' if include_rms else 'no'}-rms-mean"
            model = train_tandem_from_rows(
                train_rows,
                val_rows,
                acoustic_features=acoustic_features,
                behavioral_features=BEHAVIORAL_HEAD_FEATURES,
                vad_backend=backend,
            )
            models[variant_id] = model
            variants.append(
                {
                    "id": variant_id,
                    "vad_backend": backend,
                    "include_rms_mean": include_rms,
                    "metrics": _head_snapshot(model),
                }
            )

    hybrid_no = next(item for item in variants if item["id"] == "hybrid-no-rms-mean")
    hybrid_with = next(item for item in variants if item["id"] == "hybrid-with-rms-mean")
    drop_ok = _stacked_acc(hybrid_no["metrics"]) >= _stacked_acc(hybrid_with["metrics"]) - RMS_MEAN_HOLD
    acoustic_key = "no-rms-mean" if drop_ok else "with-rms-mean"

    hybrid = next(item for item in variants if item["id"] == f"hybrid-{acoustic_key}")
    silero = next(item for item in variants if item["id"] == f"silero-{acoustic_key}")
    if _behavioral_acc(silero["metrics"]) > _behavioral_acc(hybrid["metrics"]):
        chosen_item = silero
        backend_reason = "Silero behavioural val acc beat hybrid"
    elif _stacked_acc(silero["metrics"]) > _stacked_acc(hybrid["metrics"]):
        chosen_item = silero
        backend_reason = "Silero stacked val acc beat hybrid"
    else:
        chosen_item = hybrid
        backend_reason = "hybrid matched or beat Silero on behavioural/stacked acc"

    rms_reason = (
        f"dropped caller_rms_mean (stacked acc hold ≤ {RMS_MEAN_HOLD:.0%})"
        if drop_ok
        else "kept caller_rms_mean (stacked acc dropped more than hold)"
    )
    reason = f"{rms_reason}; {backend_reason}."
    winner = models[chosen_item["id"]]
    save_tandem(winner, output)

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "hold_rms_mean_acc": RMS_MEAN_HOLD,
        "before": before,
        "variants": variants,
        "chosen_id": chosen_item["id"],
        "reason": reason,
        "chosen": chosen_item["metrics"],
        "shipped_model": str(output),
        "revert": f"cp {BEFORE_MODEL_PATH} {output}",
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(_render_markdown(payload), encoding="utf-8")
    return winner
