"""CLI for VAD-only benchmark against organizer turn JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from is_a_human.eval.vad_benchmark import format_results, run_vad_benchmark, run_vad_benchmark_all
from is_a_human.turns.backends import VadBackendName, available_backends


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VAD backends against organizer turn JSON segments",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root with manifest.csv and turns/ (auto-detected if omitted)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max calls to evaluate")
    parser.add_argument(
        "--backend",
        choices=[*available_backends(), "all"],
        default="all",
        help="VAD backend to benchmark (default: compare all)",
    )
    args = parser.parse_args()

    if args.backend == "all":
        results = run_vad_benchmark_all(
            split=args.split,
            dataset_root=args.dataset_root,
            limit=args.limit,
        )
        print(format_results(results))
        return

    result = run_vad_benchmark(
        backend=args.backend,  # type: ignore[arg-type]
        split=args.split,
        dataset_root=args.dataset_root,
        limit=args.limit,
    )
    print(format_results([result]))
    print(f"p95_latency_ms={result.p95_latency_ms:.1f}")


if __name__ == "__main__":
    main()
