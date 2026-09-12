"""VAD-only benchmark: compare predicted speech segments to organizer JSON turns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Literal

from is_a_human.dataset.loader import DatasetError, iter_split
from is_a_human.turns.backends import VadBackendName, available_backends, get_backend
from is_a_human.turns.validation import compare_turn_segments

Split = Literal["train", "val"]


@dataclass(frozen=True)
class ChannelMetrics:
    mean_iou: float
    median_iou: float
    min_iou: float


@dataclass(frozen=True)
class VadBenchmarkResult:
    backend: VadBackendName
    split: Split
    num_calls: int
    caller: ChannelMetrics
    agent: ChannelMetrics
    mean_latency_ms: float
    p95_latency_ms: float

    @property
    def mean_overall_iou(self) -> float:
        return (self.caller.mean_iou + self.agent.mean_iou) / 2


def _channel_metrics(values: list[float]) -> ChannelMetrics:
    if not values:
        return ChannelMetrics(mean_iou=0.0, median_iou=0.0, min_iou=0.0)
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        median = sorted_values[mid]
    else:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return ChannelMetrics(
        mean_iou=mean(values),
        median_iou=median,
        min_iou=min(values),
    )


def run_vad_benchmark(
    backend: VadBackendName,
    split: Split = "val",
    dataset_root: Path | str | None = None,
    *,
    limit: int | None = None,
) -> VadBenchmarkResult:
    """Run one VAD backend on a split and score against organizer turn JSON."""
    vad = get_backend(backend)
    caller_ious: list[float] = []
    agent_ious: list[float] = []
    latencies: list[float] = []
    count = 0

    try:
        for sample in iter_split(split, root=dataset_root, load_audio=True):
            if limit is not None and count >= limit:
                break

            started = perf_counter()
            predicted = vad.detect_dual_channel(
                sample.ch0_caller,
                sample.ch1_agent,
                sample.sample_rate,
            )
            latency_ms = (perf_counter() - started) * 1000

            caller_iou = compare_turn_segments(predicted, sample.turns, channel=0).iou
            agent_iou = compare_turn_segments(predicted, sample.turns, channel=1).iou

            caller_ious.append(caller_iou)
            agent_ious.append(agent_iou)
            latencies.append(latency_ms)
            count += 1
    except DatasetError:
        return _empty_result(backend, split)

    if count == 0:
        return _empty_result(backend, split)

    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(0.95 * len(sorted_latencies)) - 1)

    return VadBenchmarkResult(
        backend=backend,
        split=split,
        num_calls=count,
        caller=_channel_metrics(caller_ious),
        agent=_channel_metrics(agent_ious),
        mean_latency_ms=mean(latencies),
        p95_latency_ms=sorted_latencies[p95_index],
    )


def run_vad_benchmark_all(
    split: Split = "val",
    dataset_root: Path | str | None = None,
    *,
    limit: int | None = None,
    backends: tuple[VadBackendName, ...] | None = None,
) -> list[VadBenchmarkResult]:
    """Run every registered VAD backend and return results in stable order."""
    names = backends or available_backends()
    return [
        run_vad_benchmark(
            backend=name,
            split=split,
            dataset_root=dataset_root,
            limit=limit,
        )
        for name in names
    ]


def _empty_result(backend: VadBackendName, split: Split) -> VadBenchmarkResult:
    empty = ChannelMetrics(mean_iou=0.0, median_iou=0.0, min_iou=0.0)
    return VadBenchmarkResult(
        backend=backend,
        split=split,
        num_calls=0,
        caller=empty,
        agent=empty,
        mean_latency_ms=0.0,
        p95_latency_ms=0.0,
    )


def format_results(results: list[VadBenchmarkResult]) -> str:
    """Render a comparison table for CLI output."""
    if not results:
        return "No results."

    lines = [
        "VAD benchmark (predicted segments vs organizer turns JSON)",
        "",
        f"{'backend':<8} {'calls':>5} {'caller_iou':>11} {'agent_iou':>10} "
        f"{'overall':>8} {'latency_ms':>11}",
        "-" * 58,
    ]
    for result in results:
        lines.append(
            f"{result.backend:<8} {result.num_calls:>5} "
            f"{result.caller.mean_iou:>11.3f} {result.agent.mean_iou:>10.3f} "
            f"{result.mean_overall_iou:>8.3f} {result.mean_latency_ms:>11.1f}"
        )
    return "\n".join(lines)
