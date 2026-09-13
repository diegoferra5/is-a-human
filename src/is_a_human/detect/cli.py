"""Train and save the tandem detector artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from is_a_human.detect.tandem import DEFAULT_MODEL_PATH, save_tandem, train_tandem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train acoustic + behavioural heads and fusion; write models/tandem.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    args = parser.parse_args()

    model = train_tandem(
        dataset_root=args.dataset_root,
        limit=args.limit,
        show_progress=args.progress,
    )
    path = save_tandem(model, args.output)
    stacked = model.metrics.get("stacked", {}).get("val", {})
    concat = model.metrics.get("concatenated", {}).get("val", {})
    print(f"fusion_type={model.fusion_type} vad={model.vad_backend}")
    print(f"stacked val auc={stacked.get('auc')} acc={stacked.get('accuracy')}")
    print(f"concat  val auc={concat.get('auc')} acc={concat.get('accuracy')}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
