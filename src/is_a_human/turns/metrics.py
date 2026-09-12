"""Conversation metrics derived from a turn ledger."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from is_a_human.turns.ledger import TurnLedger, TurnType


@dataclass(frozen=True)
class ConversationMetrics:
    caller_talk_time_s: float
    agent_talk_time_s: float
    overlap_time_s: float
    silence_time_s: float
    interruption_count: int
    mean_agent_response_latency_s: float
    mean_caller_response_latency_s: float


def _event_duration(event_type: TurnType, events, channel: int | None = None) -> float:
    total = 0.0
    for event in events:
        if event.type != event_type:
            continue
        if channel is not None and event.channel != channel:
            continue
        total += event.end - event.start
    return total


def _response_latencies(events, responder_channel: int) -> list[float]:
    latencies: list[float] = []
    other_channel = 1 - responder_channel

    stops = [
        event.end
        for event in events
        if event.type == TurnType.SPEECH and event.channel == other_channel
    ]
    starts = [
        event.start
        for event in events
        if event.type == TurnType.SPEECH and event.channel == responder_channel
    ]

    for stop_time in stops:
        next_starts = [start for start in starts if start > stop_time]
        if next_starts:
            latencies.append(next_starts[0] - stop_time)
    return latencies


def compute_conversation_metrics(ledger: TurnLedger) -> ConversationMetrics:
    """Compute foundation conversation metrics from a turn ledger."""
    events = ledger.events
    caller_talk = _event_duration(TurnType.SPEECH, events, channel=0)
    agent_talk = _event_duration(TurnType.SPEECH, events, channel=1)
    overlap = _event_duration(TurnType.OVERLAP, events)
    silence = _event_duration(TurnType.SILENCE, events)

    agent_latencies = _response_latencies(events, responder_channel=0)
    caller_latencies = _response_latencies(events, responder_channel=1)

    return ConversationMetrics(
        caller_talk_time_s=caller_talk,
        agent_talk_time_s=agent_talk,
        overlap_time_s=overlap,
        silence_time_s=silence,
        interruption_count=int(overlap > 0 and len([e for e in events if e.type == TurnType.OVERLAP])),
        mean_agent_response_latency_s=float(np.mean(agent_latencies)) if agent_latencies else 0.0,
        mean_caller_response_latency_s=float(np.mean(caller_latencies)) if caller_latencies else 0.0,
    )
