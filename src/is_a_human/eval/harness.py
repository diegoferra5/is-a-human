"""Offline evaluation harness for foundation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

from is_a_human.dataset.loader import DatasetError, iter_split
from is_a_human.pipeline import process_call
from is_a_human.turns.validation import compare_turn_segments

Split = Literal["train", "val"]


@dataclass(frozen=True)
class EvalSummary:
    split: Split
    num_calls: int
    mean_caller_iou: float
    mean_agent_iou: float
    mean_latency_ms: float
    p95_latency_ms: float


def run_foundation_eval(
    split: Split = "val",
    dataset_root: Path | str | None = None,
    *,
    limit: int | None = None,
) -> EvalSummary:
    """Run foundation pipeline on a split and compare VAD to organizer turns."""
    caller_ious: list[float] = []
    agent_ious: list[float] = []
    latencies: list[float] = []
    count = 0

    try:
        for sample in iter_split(split, root=dataset_root, load_audio=True):
            if limit is not None and count >= limit:
                break

            result = process_call(sample.ch0_caller, sample.ch1_agent, sample.sample_rate)
            predicted = result.ledger.speech_segments

            caller_iou = compare_turn_segments(predicted, sample.turns, channel=0).iou
            agent_iou = compare_turn_segments(predicted, sample.turns, channel=1).iou

            caller_ious.append(caller_iou)
            agent_ious.append(agent_iou)
            latencies.append(result.latency_ms)
            count += 1
    except DatasetError:
        return EvalSummary(
            split=split,
            num_calls=0,
            mean_caller_iou=0.0,
            mean_agent_iou=0.0,
            mean_latency_ms=0.0,
            p95_latency_ms=0.0,
        )

    if count == 0:
        return EvalSummary(
            split=split,
            num_calls=0,
            mean_caller_iou=0.0,
            mean_agent_iou=0.0,
            mean_latency_ms=0.0,
            p95_latency_ms=0.0,
        )

    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(0.95 * len(sorted_latencies)) - 1)

    return EvalSummary(
        split=split,
        num_calls=count,
        mean_caller_iou=mean(caller_ious),
        mean_agent_iou=mean(agent_ious),
        mean_latency_ms=mean(latencies),
        p95_latency_ms=sorted_latencies[p95_index],
    )
