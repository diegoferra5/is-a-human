"""End-to-end foundation pipeline for a single call."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from is_a_human.turns.ledger import TurnLedger, build_turn_ledger
from is_a_human.turns.metrics import ConversationMetrics, compute_conversation_metrics


@dataclass(frozen=True)
class PipelineResult:
    ledger: TurnLedger
    metrics: ConversationMetrics
    latency_ms: float


def process_call(
    ch0_caller: np.ndarray,
    ch1_agent: np.ndarray,
    sample_rate: int,
) -> PipelineResult:
    """Run foundation pipeline on demuxed caller/agent audio."""
    started = perf_counter()
    ledger = build_turn_ledger(ch0_caller, ch1_agent, sample_rate)
    metrics = compute_conversation_metrics(ledger)
    latency_ms = (perf_counter() - started) * 1000
    return PipelineResult(ledger=ledger, metrics=metrics, latency_ms=latency_ms)
