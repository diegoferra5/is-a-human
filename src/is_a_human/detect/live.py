"""Live tandem scoring with per-layer timings."""

from __future__ import annotations

import logging

import numpy as np

from is_a_human.analysis.features import extract_call_features_timed
from is_a_human.detect.tandem import TandemModel

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
) -> tuple[float, dict[str, float], dict[str, float | None]]:
    features, timings = extract_call_features_timed(
        anon_id=call_id,
        label="unknown",
        split="live",
        ch0_caller=ch0_caller,
        ch1_agent=ch1_agent,
        sample_rate=sample_rate,
        heavy=heavy,
        transcript_turns=transcript_turns,
    )
    probability, views = model.predict(features)
    return probability, views, timings
