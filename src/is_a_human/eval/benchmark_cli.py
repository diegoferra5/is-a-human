"""CLI: run layer benchmarks and write a Playwright-style HTML page."""

from __future__ import annotations

import argparse
from pathlib import Path

from is_a_human.eval.benchmark_html import render_benchmark_page
from is_a_human.eval.layer_benchmark import format_benchmark_text, run_layer_benchmarks

DEFAULT_OUTPUT = Path("reports/benchmarks/index.html")
ALL_SUITES = ("vad", "acoustic", "semantic", "behavioral", "tandem")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark VAD / acoustic / semantic / behavioural layers on the dataset "
            "and write a self-contained HTML report."
        )
    )
    parser.add_argument("--split", choices=["train", "val"], default="val", help="Split for VAD IoU")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--transcripts-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max calls per split")
    parser.add_argument(
        "--suites",
        default="vad,acoustic,semantic,behavioral,tandem",
        help="Comma-separated: vad,acoustic,semantic,behavioral,tandem",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    args = parser.parse_args()

    suites = tuple(part.strip() for part in args.suites.split(",") if part.strip())
    unknown = set(suites) - set(ALL_SUITES)
    if unknown:
        parser.error(f"Unknown suite(s): {', '.join(sorted(unknown))}")

    payload = run_layer_benchmarks(
        dataset_root=args.dataset_root,
        split=args.split,
        limit=args.limit,
        suites=suites,
        transcripts_dir=args.transcripts_dir,
        show_progress=args.progress,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_benchmark_page(payload), encoding="utf-8")

    print(format_benchmark_text(payload))
    print()
    print(f"Open: {args.output.resolve().as_uri()}")


if __name__ == "__main__":
    main()
