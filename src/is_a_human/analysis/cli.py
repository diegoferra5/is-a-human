import argparse
from pathlib import Path

from is_a_human.analysis.explore import (
    format_explore_report,
    format_multi_report,
    run_exploratory_analysis,
    run_multi_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exploratory analysis: compare conversational features by label"
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap-runs", type=int, default=0)
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Run train+val, bootstrap stability, and logistic baseline",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write report markdown",
    )
    args = parser.parse_args()

    if args.multi:
        summary = run_multi_analysis(
            dataset_root=args.dataset_root,
            limit=args.limit,
            bootstrap_runs=max(args.bootstrap_runs, 200),
        )
        report = format_multi_report(summary)
    else:
        summary = run_exploratory_analysis(
            split=args.split,
            dataset_root=args.dataset_root,
            limit=args.limit,
            bootstrap_runs=args.bootstrap_runs,
        )
        report = format_explore_report(summary)

    print(report)
    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")
