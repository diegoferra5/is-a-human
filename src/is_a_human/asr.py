"""Caller-channel transcription for the semantic head.

Only the stretches the VAD marked as caller speech are sent to Whisper, joined
with a short pad. On this data that is roughly a third of the call, so the slow
step costs a third of what a whole-channel pass would. Times are mapped back
to call time so the turns sort the same way as offline transcripts.
"""

from __future__ import annotations

import numpy as np

MODEL = "mlx-community/whisper-large-v3-turbo"
TARGET_SR = 16000
PAD_S = 0.15


def _to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SR:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly
    g = int(np.gcd(sample_rate, TARGET_SR))
    return resample_poly(audio, TARGET_SR // g, sample_rate // g).astype(np.float32)


def _whisper_segments(audio_16k: np.ndarray, model: str) -> list[dict]:
    import mlx_whisper
    result = mlx_whisper.transcribe(
        audio_16k,
        path_or_hf_repo=model,
        language="es",
        condition_on_previous_text=False,   # phone audio: stops hallucination loops
        temperature=0.0,
        verbose=False,
    )
    return [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()}
        for seg in result["segments"]
        if seg["text"].strip()
    ]


def transcribe_caller(
    ch0_caller: np.ndarray,
    sample_rate: int,
    caller_segments: list[tuple[float, float]],
    *,
    model: str = MODEL,
    pad_s: float = PAD_S,
) -> list[dict]:
    """VAD-trimmed caller audio -> [{channel: 0, start, end, text}] in call time."""
    if not caller_segments:
        return []
    n = len(ch0_caller)
    pieces: list[np.ndarray] = []
    offsets: list[tuple[float, float]] = []     # (start in joined audio, start in call)
    cursor = 0.0
    for start, end in sorted(caller_segments):
        a = max(0, int((start - pad_s) * sample_rate))
        b = min(n, int((end + pad_s) * sample_rate))
        if b <= a:
            continue
        chunk = ch0_caller[a:b]
        pieces.append(chunk)
        offsets.append((cursor, a / sample_rate))
        cursor += len(chunk) / sample_rate
    if not pieces:
        return []
    joined = np.concatenate(pieces)
    segs = _whisper_segments(_to_16k(joined, sample_rate), model)

    def to_call_time(t: float) -> float:
        base_joined, base_call = offsets[0]
        for j, c in offsets:
            if j <= t:
                base_joined, base_call = j, c
        return round(base_call + (t - base_joined), 2)

    return [
        {"channel": 0, "start": to_call_time(s["start"]), "end": to_call_time(s["end"]), "text": s["text"]}
        for s in segs
    ]
