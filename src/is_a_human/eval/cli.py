import argparse
from pathlib import Path

from is_a_human.eval.harness import run_foundation_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run foundation pipeline eval on a dataset split")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root with manifest.csv and turns/ (auto-detected if omitted)",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = run_foundation_eval(
        split=args.split,
        dataset_root=args.dataset_root,
        limit=args.limit,
    )

    print(f"split={summary.split} calls={summary.num_calls}")
    print(f"mean_caller_iou={summary.mean_caller_iou:.3f}")
    print(f"mean_agent_iou={summary.mean_agent_iou:.3f}")
    print(f"mean_latency_ms={summary.mean_latency_ms:.1f}")
    print(f"p95_latency_ms={summary.p95_latency_ms:.1f}")
