"""Train and save the tandem detector artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from is_a_human.detect.tandem import DEFAULT_MODEL_PATH, save_tandem, train_tandem
from is_a_human.detect.tune import REPORT_MD, run_head_tune
from is_a_human.turns.backends import DEFAULT_VAD_BACKEND, available_backends


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train acoustic + behavioural heads and fusion; write models/tandem.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("transcripts"),
        help="Whisper transcripts <anon_id>.json; enables the semantic head (pass '' to disable)",
    )
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument(
        "--vad-backend",
        choices=available_backends(),
        default=DEFAULT_VAD_BACKEND,
        help="VAD used when not running --tune",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Ablate caller_rms_mean and compare hybrid vs Silero; write reports/tune/",
    )
    args = parser.parse_args()

    if args.tune:
        model = run_head_tune(
            dataset_root=args.dataset_root,
            limit=args.limit,
            show_progress=args.progress,
            output=args.output,
        )
        report = REPORT_MD.resolve()
    else:
        transcripts = args.transcripts_dir if str(args.transcripts_dir) else None
        model = train_tandem(
            dataset_root=args.dataset_root,
            limit=args.limit,
            show_progress=args.progress,
            vad_backend=args.vad_backend,
            transcripts_dir=transcripts,
        )
        save_tandem(model, args.output)
        report = None

    stacked = model.metrics.get("stacked", {}).get("val", {})
    concat = model.metrics.get("concatenated", {}).get("val", {})
    print(f"fusion_type={model.fusion_type} vad={model.vad_backend}")
    print(f"stacked val auc={stacked.get('auc')} acc={stacked.get('accuracy')}")
    print(f"concat  val auc={concat.get('auc')} acc={concat.get('accuracy')}")
    sem = model.metrics.get("semantic", {}).get("val", {})
    s3 = model.metrics.get("stacked3", {}).get("val", {})
    gate = model.metrics.get("gate", {})
    if sem:
        print(f"semantic val auc={sem.get('auc')} acc={sem.get('accuracy')}")
        print(f"3-head gated val acc={s3.get('gated', {}).get('accuracy')} "
              f"(always={s3.get('always', {}).get('accuracy')}, gated {s3.get('gated_fraction')})")
        print(f"gate margin={gate.get('margin')} train-oof gated={gate.get('train_oof_gated_fraction')} "
              f"misses inside={gate.get('train_oof_misses_inside_gate')}/{gate.get('train_oof_fast_misses')}")
    print(f"saved {args.output}")
    if report is not None:
        print(f"report {report}")


if __name__ == "__main__":
    main()
