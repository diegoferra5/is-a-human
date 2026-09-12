"""Transcribe the challenge calls with mlx-whisper (Apple Silicon).

Each call is a stereo 8 kHz WAV: channel 0 = caller, channel 1 = agent.
We split the channels, upsample to 16 kHz (Whisper's native rate), and
transcribe each side separately.

Output, one JSON per call, same shape as the dataset's turns/ files plus text:
    {"call_id": ..., "turns": [{"channel": 0, "start": 10.5, "end": 20.2, "text": "..."}]}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx_whisper
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL = "mlx-community/whisper-large-v3-turbo"
TARGET_SR = 16000


def load_channel(wav_path: Path, channel: int) -> np.ndarray:
    """One channel as float32 at 16 kHz."""
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = audio[:, channel]
    if sr != TARGET_SR:
        up, down = TARGET_SR // np.gcd(sr, TARGET_SR), sr // np.gcd(sr, TARGET_SR)
        mono = resample_poly(mono, up, down).astype(np.float32)
    return mono


def transcribe_channel(audio: np.ndarray, model: str) -> list[dict]:
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model,
        language="es",
        condition_on_previous_text=False,  # phone audio: stops hallucination loops
        temperature=0.0,
        verbose=False,
    )
    return [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
        if seg["text"].strip()
    ]


def transcribe_call(wav_path: Path, out_dir: Path, channels: list[int], model: str) -> dict:
    call_id = wav_path.stem
    turns = []
    for ch in channels:
        audio = load_channel(wav_path, ch)
        for seg in transcribe_channel(audio, model):
            turns.append({"channel": ch, **seg})
    turns.sort(key=lambda t: t["start"])

    out = {"call_id": call_id, "turns": turns}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{call_id}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=Path("dataset/audio"))
    ap.add_argument("--out-dir", type=Path, default=Path("transcripts"))
    ap.add_argument("--channels", default="0,1", help="which channels, e.g. '1' for agent only")
    ap.add_argument("--limit", type=int, default=0, help="only the first N calls")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--force", action="store_true", help="re-transcribe calls already done")
    args = ap.parse_args()

    channels = [int(c) for c in args.channels.split(",")]
    wavs = sorted(args.audio_dir.glob("*.wav"))
    if args.limit:
        wavs = wavs[: args.limit]

    todo = wavs if args.force else [w for w in wavs if not (args.out_dir / f"{w.stem}.json").exists()]
    print(f"{len(wavs)} calls, {len(todo)} to do, channels={channels}", flush=True)

    t0 = time.time()
    for i, wav in enumerate(todo, 1):
        try:
            out = transcribe_call(wav, args.out_dir, channels, args.model)
            elapsed = time.time() - t0
            eta = elapsed / i * (len(todo) - i)
            print(
                f"[{i}/{len(todo)}] {wav.stem}  {len(out['turns'])} segs  "
                f"{elapsed/60:.1f}m elapsed, ~{eta/60:.0f}m left",
                flush=True,
            )
        except Exception as exc:  # keep going; one bad file shouldn't kill the run
            print(f"[{i}/{len(todo)}] {wav.stem} FAILED: {exc}", file=sys.stderr, flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
