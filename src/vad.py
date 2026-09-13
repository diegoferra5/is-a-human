"""Voice Activity Detection: raw stereo WAV -> per-channel speech turns.

At judging time we receive only a WAV (no turns.json), so we must reproduce
the {channel, start, end} segments ourselves. This is an energy-based VAD:
frame the signal, measure loudness, threshold against the per-channel noise
floor, then merge frames into segments. No external deps (stdlib wave + numpy).

Output matches the challenge format:
    {"turns": [{"channel": 0, "start": 0.96, "end": 1.5}, ...]}
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

FRAME_MS = 25
HOP_MS = 10
MIN_SPEECH_MS = 120      # drop blips shorter than this
MIN_GAP_MS = 200         # bridge silences shorter than this (same turn)
THRESH_DB = 6.0          # frame is speech if this many dB above the noise floor


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (samples[n, channels] float32 in [-1,1], sample_rate)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    assert width == 2, f"expected 16-bit PCM, got {width*8}-bit"
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    x = x.reshape(-1, n_ch)
   ## print(x,sr)
    return x, sr

## this detects the energy of chunk
def _frame_energy_db(sig: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    frame = int(sr * FRAME_MS / 1000)
    hop = int(sr * HOP_MS / 1000)
    if len(sig) < frame:
        return np.array([]), hop
    n = 1 + (len(sig) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = sig[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    return 20 * np.log10(rms + 1e-10), hop


def _segments(mask: np.ndarray, hop: int, sr: int) -> list[tuple[float, float]]:
    """Boolean speech-mask over frames -> merged (start,end) in seconds."""
    hop_s = hop / sr
    frame_s = FRAME_MS / 1000
    segs = []
    in_seg = False
    for i, m in enumerate(mask):
        t = i * hop_s
        if m and not in_seg:
            start, in_seg = t, True
        elif not m and in_seg:
            segs.append((start, t + frame_s)); in_seg = False
    if in_seg:
        segs.append((start, len(mask) * hop_s + frame_s))

    # bridge short gaps
    merged = []
    for s, e in segs:
        if merged and s - merged[-1][1] < MIN_GAP_MS / 1000:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # drop short blips
    return [(s, e) for s, e in merged if (e - s) >= MIN_SPEECH_MS / 1000]


def channel_turns(sig: np.ndarray, sr: int) -> list[tuple[float, float]]:
    db, hop = _frame_energy_db(sig, sr)
    if db.size == 0:
        return []
    # noise floor = low percentile of frame energies; threshold above it.
    floor = np.percentile(db, 10)
    mask = db > (floor + THRESH_DB)
    return _segments(mask, hop, sr)


# ---- Silero VAD backend (pretrained neural net; sharper boundaries) ---------
# Small model (~1MB) that natively supports 8 kHz. We load it once and reuse it.
_silero_model = None


def _load_silero():
    global _silero_model
    if _silero_model is None:
        from silero_vad import load_silero_vad
        _silero_model = load_silero_vad()
    return _silero_model


def channel_turns_silero(sig: np.ndarray, sr: int) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps
    model = _load_silero()
    audio = torch.from_numpy(sig.astype(np.float32))
    ts = get_speech_timestamps(audio, model, sampling_rate=sr, return_seconds=True)
    return [(seg["start"], seg["end"]) for seg in ts]


def extract_turns(path: str | Path, backend: str = "silero") -> dict:
    """backend="silero" (default, accurate) or "energy" (simple fallback)."""
    x, sr = read_wav(path)
    turns = []
    for ch in range(x.shape[1]):
        if backend == "silero":
            segs = channel_turns_silero(x[:, ch], sr)
        else:
            segs = channel_turns(x[:, ch], sr)
        for s, e in segs:
            turns.append({"channel": ch, "start": round(s, 2), "end": round(e, 2)})
    turns.sort(key=lambda t: t["start"])
    return {"turns": turns}


if __name__ == "__main__":
    import sys, json
    print(json.dumps(extract_turns(sys.argv[1]), indent=0)[:600])
