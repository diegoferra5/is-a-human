"""In-memory stereo WAV demuxing for telephony audio."""

from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
import soundfile as sf

from is_a_human.audio.errors import (
    InvalidBase64Error,
    InvalidWavError,
    NotStereoError,
    SampleRateError,
)

SUPPORTED_SAMPLE_RATES = {8000, 16000}


def demux_base64_telephony(base64_payload: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Decode base64 WAV and split into caller (ch0) and agent (ch1) mono arrays."""
    try:
        raw_bytes = base64.b64decode(base64_payload, validate=True)
    except Exception as exc:
        raise InvalidBase64Error("Failed to decode base64 audio payload") from exc
    return demux_wav_bytes(raw_bytes)


def demux_wav_bytes(raw_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, int]:
    """Decode WAV bytes and split into caller (ch0) and agent (ch1) mono arrays."""
    buffer = io.BytesIO(raw_bytes)
    try:
        data, sample_rate = sf.read(buffer, dtype="float32", always_2d=True)
    except Exception as exc:
        raise InvalidWavError("Failed to read WAV audio from bytes") from exc

    if data.shape[1] < 2:
        raise NotStereoError(
            f"Expected stereo audio with at least 2 channels, got shape {data.shape}"
        )

    if int(sample_rate) not in SUPPORTED_SAMPLE_RATES:
        raise SampleRateError(
            f"Unsupported sample rate {sample_rate} Hz; expected one of {sorted(SUPPORTED_SAMPLE_RATES)}"
        )

    ch0_caller = np.ascontiguousarray(data[:, 0])
    ch1_agent = np.ascontiguousarray(data[:, 1])
    return ch0_caller, ch1_agent, int(sample_rate)


def load_stereo_wav(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load a stereo WAV file from disk into caller/agent mono arrays."""
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise InvalidWavError(f"Failed to read WAV file: {path}") from exc

    if data.shape[1] < 2:
        raise NotStereoError(
            f"Expected stereo audio with at least 2 channels, got shape {data.shape}"
        )

    if int(sample_rate) not in SUPPORTED_SAMPLE_RATES:
        raise SampleRateError(
            f"Unsupported sample rate {sample_rate} Hz; expected one of {sorted(SUPPORTED_SAMPLE_RATES)}"
        )

    ch0_caller = np.ascontiguousarray(data[:, 0])
    ch1_agent = np.ascontiguousarray(data[:, 1])
    return ch0_caller, ch1_agent, int(sample_rate)
