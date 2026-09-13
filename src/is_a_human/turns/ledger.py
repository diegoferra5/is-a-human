"""Turn ledger construction from dual-channel VAD."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from is_a_human.turns.backends import VadBackend, VadBackendName, get_backend
from is_a_human.turns.segments import VadSegment
from is_a_human.turns.vad import detect_dual_channel_segments, segments_to_mask


class TurnType(str, Enum):
    SPEECH = "speech"
    OVERLAP = "overlap"
    SILENCE = "silence"


@dataclass(frozen=True)
class TurnEvent:
    channel: int | None
    start: float
    end: float
    type: TurnType


@dataclass(frozen=True)
class TurnLedger:
    events: tuple[TurnEvent, ...]
    speech_segments: tuple[VadSegment, ...]
    frame_duration_s: float


def _mask_to_events(
    ch0_mask: np.ndarray,
    ch1_mask: np.ndarray,
    frame_duration_s: float,
) -> list[TurnEvent]:
    events: list[TurnEvent] = []
    if ch0_mask.size == 0:
        return events

    current_type: TurnType | None = None
    current_channel: int | None = None
    start_idx = 0

    for idx in range(ch0_mask.size):
        # Overlap takes precedence: both channels active in the same frame.
        if ch0_mask[idx] and ch1_mask[idx]:
            frame_type = TurnType.OVERLAP
            frame_channel = None
        elif ch0_mask[idx]:
            frame_type = TurnType.SPEECH
            frame_channel = 0
        elif ch1_mask[idx]:
            frame_type = TurnType.SPEECH
            frame_channel = 1
        else:
            frame_type = TurnType.SILENCE
            frame_channel = None

        if current_type is None:
            current_type = frame_type
            current_channel = frame_channel
            start_idx = idx
            continue

        if frame_type != current_type or frame_channel != current_channel:
            events.append(
                TurnEvent(
                    channel=current_channel,
                    start=start_idx * frame_duration_s,
                    end=idx * frame_duration_s,
                    type=current_type,
                )
            )
            current_type = frame_type
            current_channel = frame_channel
            start_idx = idx

    events.append(
        TurnEvent(
            channel=current_channel,
            start=start_idx * frame_duration_s,
            end=ch0_mask.size * frame_duration_s,
            type=current_type or TurnType.SILENCE,
        )
    )
    return events


def build_turn_ledger(
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
    *,
    frame_duration_s: float = 0.032,
    backend: VadBackend | VadBackendName | None = None,
) -> TurnLedger:
    """Build an aligned turn ledger from dual-channel VAD."""
    duration_s = max(len(ch0_caller), len(ch1_agent)) / sample_rate
    speech_segments = detect_dual_channel_segments(
        ch0_caller, ch1_agent, sample_rate, backend=backend or get_backend()
    )

    caller_segments = [segment for segment in speech_segments if segment.channel == 0]
    agent_segments = [segment for segment in speech_segments if segment.channel == 1]

    ch0_mask = segments_to_mask(
        caller_segments, duration_s, sample_rate, frame_duration_s=frame_duration_s
    )
    ch1_mask = segments_to_mask(
        agent_segments, duration_s, sample_rate, frame_duration_s=frame_duration_s
    )

    num_frames = max(ch0_mask.size, ch1_mask.size)
    if ch0_mask.size < num_frames:
        ch0_mask = np.pad(ch0_mask, (0, num_frames - ch0_mask.size))
    if ch1_mask.size < num_frames:
        ch1_mask = np.pad(ch1_mask, (0, num_frames - ch1_mask.size))

    events = tuple(_mask_to_events(ch0_mask, ch1_mask, frame_duration_s))
    return TurnLedger(
        events=events,
        speech_segments=tuple(speech_segments),
        frame_duration_s=frame_duration_s,
    )


def speech_segments_from_ledger(ledger: TurnLedger) -> Sequence[VadSegment]:
    return ledger.speech_segments
