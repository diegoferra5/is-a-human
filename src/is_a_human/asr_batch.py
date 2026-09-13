"""Offline transcription with the SAME trimming the live endpoint uses.

The semantic head must see at training time exactly what it sees at serve
time: the caller's VAD segments joined and transcribed, not the whole channel.
Writes <out_dir>/<anon_id>.json = {"call_id", "turns", "caller_speech_s"}.
Resumable; re-run after an interruption and it picks up where it stopped.

    python -m is_a_human.asr_batch --dataset-root dataset --out transcripts_trimmed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from is_a_human.asr import transcribe_caller
from is_a_human.dataset.loader import iter_split
from is_a_human.pipeline import process_call
from is_a_human.turns.backends import DEFAULT_VAD_BACKEND


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="dataset")
    ap.add_argument("--out", type=Path, default=Path("transcripts_trimmed"))
    ap.add_argument("--vad-backend", default=DEFAULT_VAD_BACKEND)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    done = n = 0
    started = time.time()
    for split in ("train", "val"):
        for sample in iter_split(split, root=args.dataset_root, load_audio=True):
            n += 1
            target = args.out / f"{sample.anon_id}.json"
            if target.exists() and target.stat().st_size > 0:
                continue
            result = process_call(sample.ch0_caller, sample.ch1_agent, sample.sample_rate, backend=args.vad_backend)
            segs = [(s.start, s.end) for s in result.ledger.speech_segments if s.channel == 0]
            turns = transcribe_caller(sample.ch0_caller, sample.sample_rate, segs)
            payload = {"call_id": sample.anon_id, "turns": turns,
                       "caller_speech_s": round(sum(e - s for s, e in segs), 2)}
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
            done += 1
            if done % 25 == 0:
                print(f"{done} transcribed, {n} seen, {time.time() - started:.0f}s", flush=True)
    print(f"finished: {done} new, {n} total, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
