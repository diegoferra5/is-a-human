"""Live tandem scoring with per-layer timings."""

from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter
from typing import Callable

import numpy as np

from is_a_human.analysis.features import extract_call_features_timed
from is_a_human.analysis.semantic import extract_semantic_features
from is_a_human.detect.tandem import TandemModel
from is_a_human.pipeline import process_call

Transcriber = Callable[[np.ndarray, int, list[tuple[float, float]]], list[dict]]

logger = logging.getLogger("is_a_human.detect")


def format_layer_ms(value: float | None) -> str:
    return "skip" if value is None else f"{value:.1f}ms"


def log_layer_timings(call_id: str, timings: dict[str, float | None], *, total_ms: float) -> None:
    logger.info(
        "layers call_id=%s vad=%s acoustic=%s semantic=%s behavioral=%s extras=%s total=%.1fms",
        call_id,
        format_layer_ms(timings.get("vad")),
        format_layer_ms(timings.get("acoustic")),
        format_layer_ms(timings.get("semantic")),
        format_layer_ms(timings.get("behavioral")),
        format_layer_ms(timings.get("acoustic_extras")),
        total_ms,
    )


def detect_from_audio(
    model: TandemModel,
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    *,
    call_id: str = "live",
    heavy: bool = False,
    transcript_turns: list[dict] | None = None,
    vad_backend: str | None = None,
    transcribe: Transcriber | None = None,
) -> tuple[float, dict[str, float], dict[str, float | None]]:
    """Fast path first. If the two-head fusion lands inside the gate and the
    model has a semantic head, transcribe the caller's VAD segments (or use the
    transcript_turns given) and let the three-head fusion decide."""
    backend = vad_backend if vad_backend is not None else model.vad_backend
    started = perf_counter()
    result = process_call(ch0_caller, ch1_agent, sample_rate, backend=backend)
    vad_ms = (perf_counter() - started) * 1000

    features, timings = extract_call_features_timed(
        anon_id=call_id,
        label="unknown",
        split="live",
        ch0_caller=ch0_caller,
        ch1_agent=ch1_agent,
        sample_rate=sample_rate,
        pipeline_result=result,
        heavy=heavy,
        transcript_turns=transcript_turns,
        vad_backend=backend,
    )
    timings["vad"] = vad_ms
    probability, views = model.predict(features)
    timings["gated"] = 0.0

    availability = model.head_availability(features)
    if model.needs_semantic(probability, availability, views):
        timings["gated"] = 1.0
        if transcript_turns is None and transcribe is not None:
            started = perf_counter()
            caller_segments = [(seg.start, seg.end) for seg in result.ledger.speech_segments if seg.channel == 0]
            if not caller_segments:
                # VAD found nothing: the fast heads are already masked, so the
                # semantic head is all we have -- give Whisper the whole channel.
                caller_segments = [(0.0, len(ch0_caller) / sample_rate)]
            transcript_turns = transcribe(ch0_caller, sample_rate, caller_segments)
            caller_speech_s = sum(e - s for s, e in caller_segments)
            semantic = extract_semantic_features(transcript_turns, caller_speech_s=caller_speech_s)
            features = replace(features, **semantic)
            timings["semantic"] = (perf_counter() - started) * 1000
        probability, views = model.predict_full(features)
    return probability, views, timings
