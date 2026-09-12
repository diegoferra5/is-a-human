"""Pluggable VAD backends: Silero, energy, and hybrid (energy + Silero fallback)."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from is_a_human.turns.energy_vad import detect_energy_segments
from is_a_human.turns.segments import VadSegment
from is_a_human.turns.silero_vad import detect_silero_segments

VadBackendName = Literal["silero", "energy", "hybrid"]

DEFAULT_VAD_BACKEND: VadBackendName = "hybrid"


class VadBackend(Protocol):
    name: str

    def detect_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        channel: int,
    ) -> list[VadSegment]: ...

    def detect_dual_channel(
        self,
        ch0_caller: np.ndarray,
        ch1_agent: np.ndarray,
        sample_rate: int,
    ) -> list[VadSegment]: ...


class SileroBackend:
    name = "silero"

    def detect_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        channel: int,
    ) -> list[VadSegment]:
        return detect_silero_segments(audio, sample_rate, channel=channel)

    def detect_dual_channel(
        self,
        ch0_caller: np.ndarray,
        ch1_agent: np.ndarray,
        sample_rate: int,
    ) -> list[VadSegment]:
        return (
            self.detect_segments(ch0_caller, sample_rate, channel=0)
            + self.detect_segments(ch1_agent, sample_rate, channel=1)
        )


class EnergyBackend:
    name = "energy"

    def detect_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        channel: int,
    ) -> list[VadSegment]:
        return detect_energy_segments(audio, sample_rate, channel=channel)

    def detect_dual_channel(
        self,
        ch0_caller: np.ndarray,
        ch1_agent: np.ndarray,
        sample_rate: int,
    ) -> list[VadSegment]:
        return (
            self.detect_segments(ch0_caller, sample_rate, channel=0)
            + self.detect_segments(ch1_agent, sample_rate, channel=1)
        )


class HybridBackend:
    """Energy VAD fitted to organizer turns; Silero timestamps if energy is empty.

    Organizer JSON matches energy VAD on this telephony audio. Hard-AND with
    Silero probabilities drops true caller speech. Use energy as the detector
    and Silero's timestamp API only as a fallback.
    """

    name = "hybrid"

    def detect_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        channel: int,
    ) -> list[VadSegment]:
        segments = detect_energy_segments(audio, sample_rate, channel=channel)
        if segments:
            return segments
        return detect_silero_segments(audio, sample_rate, channel=channel)

    def detect_dual_channel(
        self,
        ch0_caller: np.ndarray,
        ch1_agent: np.ndarray,
        sample_rate: int,
    ) -> list[VadSegment]:
        return (
            self.detect_segments(ch0_caller, sample_rate, channel=0)
            + self.detect_segments(ch1_agent, sample_rate, channel=1)
        )


_BACKENDS: dict[VadBackendName, VadBackend] = {
    "silero": SileroBackend(),
    "energy": EnergyBackend(),
    "hybrid": HybridBackend(),
}


def get_backend(name: VadBackendName | None = None) -> VadBackend:
    return _BACKENDS[name or DEFAULT_VAD_BACKEND]


def available_backends() -> tuple[VadBackendName, ...]:
    return tuple(_BACKENDS.keys())
