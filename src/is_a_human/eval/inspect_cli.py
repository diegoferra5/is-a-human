"""CLI: score a split with the shipped tandem and write miss/disagreement tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from is_a_human.detect.tandem import DEFAULT_MODEL_PATH
from is_a_human.eval.inspect import DEFAULT_JSON, DEFAULT_MD, run_val_inspect


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect tandem misses and head disagreement on a dataset split."
    )
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    args = parser.parse_args()

    payload = run_val_inspect(
        model_path=args.model,
        dataset_root=args.dataset_root,
        split=args.split,
        limit=args.limit,
        show_progress=args.progress,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    summary = payload["summary"]
    print(
        f"split={payload['split']} acc={summary['accuracy']} "
        f"balanced={summary['balanced_accuracy']} misses={summary['n_miss']}/{summary['calls']}"
    )
    for name, part in (payload.get("by_split") or {}).items():
        print(
            f"  {name}: acc={part['accuracy']} balanced={part['balanced_accuracy']} "
            f"misses={part['n_miss']}/{part['calls']}"
        )
    print(f"miss_agree={summary['kinds']['miss_agree']} miss_mixer={summary['kinds']['miss_mixer']}")
    for row in payload["misses"]:
        print(
            f"  {row['anon_id']} label={row['label']} fused={row['p_fused']:.3f} "
            f"ac={row['p_acoustic']:.3f} beh={row['p_behavioral']:.3f} {row['error_kind']}"
        )
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
