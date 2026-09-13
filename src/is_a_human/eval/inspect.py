"""Per-call tandem inspection: misses, head disagreement, balanced accuracy."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from is_a_human.analysis.explore import _collect_features
from is_a_human.analysis.features import ACOUSTIC_HEAD_FEATURES, BEHAVIORAL_HEAD_FEATURES, CallFeatures
from is_a_human.detect.tandem import DEFAULT_MODEL_PATH, TandemModel, load_tandem

DEFAULT_JSON = Path("reports/tune/val-inspect.json")
DEFAULT_MD = Path("reports/tune/val-inspect.md")


def _verdict(probability: float) -> str:
    return "synthetic" if probability >= 0.5 else "human"


def _error_kind(label: str, fused: str, acoustic: str, behavioral: str) -> str:
    if fused == label:
        if acoustic != behavioral:
            return "ok_disagree"
        return "ok"
    if acoustic == behavioral:
        return "miss_agree"
    return "miss_mixer"


def score_call(model: TandemModel, row: CallFeatures) -> dict:
    probability, views = model.predict(row)
    fused = _verdict(probability)
    acoustic = _verdict(views["acoustic"])
    behavioral = _verdict(views["behavioral"])
    return {
        "anon_id": row.anon_id,
        "label": row.label,
        "split": row.split,
        "p_fused": round(probability, 4),
        "p_acoustic": round(views["acoustic"], 4),
        "p_behavioral": round(views["behavioral"], 4),
        "pred_fused": fused,
        "pred_acoustic": acoustic,
        "pred_behavioral": behavioral,
        "error_kind": _error_kind(row.label, fused, acoustic, behavioral),
        "behavioral_features": {
            name: round(float(getattr(row, name)), 4) for name in BEHAVIORAL_HEAD_FEATURES
        },
        "acoustic_features": {
            name: round(float(getattr(row, name)), 4) for name in ACOUSTIC_HEAD_FEATURES
        },
    }


def summarize_scores(rows: list[dict]) -> dict:
    n = len(rows)
    n_syn = sum(1 for row in rows if row["label"] == "synthetic")
    n_hum = sum(1 for row in rows if row["label"] == "human")
    tp = sum(1 for row in rows if row["label"] == "synthetic" and row["pred_fused"] == "synthetic")
    tn = sum(1 for row in rows if row["label"] == "human" and row["pred_fused"] == "human")
    tpr = tp / n_syn if n_syn else None
    tnr = tn / n_hum if n_hum else None
    kinds = {
        "ok": sum(1 for row in rows if row["error_kind"] == "ok"),
        "ok_disagree": sum(1 for row in rows if row["error_kind"] == "ok_disagree"),
        "miss_agree": sum(1 for row in rows if row["error_kind"] == "miss_agree"),
        "miss_mixer": sum(1 for row in rows if row["error_kind"] == "miss_mixer"),
    }
    return {
        "calls": n,
        "n_synthetic": n_syn,
        "n_human": n_hum,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "tpr_synthetic": round(tpr, 4) if tpr is not None else None,
        "tnr_human": round(tnr, 4) if tnr is not None else None,
        "balanced_accuracy": round((tpr + tnr) / 2, 4) if tpr is not None and tnr is not None else None,
        "kinds": kinds,
        "n_miss": kinds["miss_agree"] + kinds["miss_mixer"],
    }


def inspect_rows(model: TandemModel, rows: list[CallFeatures]) -> dict:
    scored = [score_call(model, row) for row in rows]
    misses = [row for row in scored if row["error_kind"].startswith("miss")]
    disagreements = [row for row in scored if row["pred_acoustic"] != row["pred_behavioral"]]
    splits = sorted({row["split"] for row in scored})
    by_split = {
        name: summarize_scores([row for row in scored if row["split"] == name])
        for name in splits
    }
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "fusion_type": model.fusion_type,
        "vad_backend": model.vad_backend,
        "summary": summarize_scores(scored),
        "by_split": by_split,
        "misses": misses,
        "disagreements": disagreements,
        "calls": scored,
    }


def render_inspect_markdown(payload: dict) -> str:
    summary = payload["summary"]
    split = payload.get("split", "val")
    lines = [
        f"# Tandem inspection ({split})",
        "",
        f"Generated {payload['generated_at']}. fusion `{payload['fusion_type']}` · VAD `{payload['vad_backend']}`.",
        "",
        f"- calls {summary['calls']} (synthetic {summary['n_synthetic']} / human {summary['n_human']})",
        f"- accuracy **{summary['accuracy']}** · balanced acc **{summary['balanced_accuracy']}**",
        f"- TPR (synthetic) {summary['tpr_synthetic']} · TNR (human) {summary['tnr_human']}",
        f"- misses **{summary['n_miss']}** "
        f"(agree {summary['kinds']['miss_agree']} · mixer picked wrong head {summary['kinds']['miss_mixer']})",
        f"- heads disagree {summary['kinds']['ok_disagree'] + summary['kinds']['miss_mixer']} "
        f"(mixer still right on {summary['kinds']['ok_disagree']})",
        "",
    ]
    by_split = payload.get("by_split") or {}
    if by_split:
        lines.append("## By split")
        lines.append("")
        for name, part in by_split.items():
            lines.append(
                f"- **{name}** acc {part['accuracy']} · balanced {part['balanced_accuracy']} "
                f"· misses {part['n_miss']} / {part['calls']}"
            )
        lines.extend(["", "## Misses", ""])
    else:
        lines.extend(["## Misses", ""])
    if not payload["misses"]:
        lines.append("None.")
    else:
        lines.append(
            "| call | split | label | fused p | acoustic p | behavioural p | kind | pos_median | agent_talk | recovery_cv |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in payload["misses"]:
            beh = row["behavioral_features"]
            lines.append(
                f"| `{row['anon_id']}` | {row['split']} | {row['label']} | {row['p_fused']:.3f} | "
                f"{row['p_acoustic']:.3f} | {row['p_behavioral']:.3f} | {row['error_kind']} | "
                f"{beh['caller_response_latency_pos_median_s']:.3f} | "
                f"{beh['agent_talk_ratio']:.3f} | {beh['agent_aligned_recovery_cv']:.3f} |"
            )
        lines.append("")
        lines.append("Acoustic head on misses:")
        lines.append("")
        lines.append("| call | rms_cv | zcr_std | crest_cv | flatness_std | centroid_std |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in payload["misses"]:
            ac = row["acoustic_features"]
            lines.append(
                f"| `{row['anon_id']}` | {ac['caller_rms_cv']:.3f} | {ac['caller_zcr_std']:.3f} | "
                f"{ac['caller_crest_factor_cv']:.3f} | {ac['caller_spectral_flatness_std']:.3f} | "
                f"{ac['caller_spectral_centroid_std']:.1f} |"
            )
    lines.extend(["", "## How to read `error_kind`", ""])
    lines.extend(
        [
            "- `miss_agree` — both heads on the wrong side of 0.5; a new feature is the only fix.",
            "- `miss_mixer` — heads disagree and fusion followed the wrong one; threshold/mixer, not a new layer.",
            "- `ok_disagree` — heads disagree but fusion followed the right one.",
            "",
        ]
    )
    return "\n".join(lines)


def run_val_inspect(
    *,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    dataset_root: Path | str | None = None,
    split: str = "val",
    limit: int | None = None,
    show_progress: bool = True,
    output_json: Path = DEFAULT_JSON,
    output_md: Path = DEFAULT_MD,
) -> dict:
    model = load_tandem(model_path)
    splits = ("train", "val") if split == "all" else (split,)
    rows: list[CallFeatures] = []
    remaining = limit
    for part in splits:
        part_limit = remaining
        part_rows = _collect_features(
            part,  # type: ignore[arg-type]
            dataset_root,
            limit=part_limit,
            show_progress=show_progress,
            heavy=False,
            vad_backend=model.vad_backend,
        )
        rows.extend(part_rows)
        if remaining is not None:
            remaining -= len(part_rows)
            if remaining <= 0:
                break
    payload = inspect_rows(model, rows)
    payload["model_path"] = str(model_path)
    payload["split"] = split
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_inspect_markdown(payload), encoding="utf-8")
    return payload
