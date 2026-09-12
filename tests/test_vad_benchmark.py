import pytest

from is_a_human.eval.vad_benchmark import run_vad_benchmark, run_vad_benchmark_all
from is_a_human.turns.backends import available_backends

pytestmark = pytest.mark.vad


@pytest.mark.integration
@pytest.mark.parametrize("backend", available_backends())
def test_vad_benchmark_backend_on_val_subset(dataset_paths, backend, report_metrics):
    result = run_vad_benchmark(
        backend=backend,
        split="val",
        dataset_root=dataset_paths.root,
        limit=3,
    )
    report_metrics(
        backend=backend,
        caller_iou=result.caller.mean_iou,
        agent_iou=result.agent.mean_iou,
        overall_iou=result.mean_overall_iou,
        latency_ms=result.mean_latency_ms,
    )

    assert result.num_calls == 3
    assert result.backend == backend
    assert result.caller.mean_iou > 0.3
    assert result.agent.mean_iou > 0.3
    assert result.mean_latency_ms > 0


@pytest.mark.integration
def test_vad_benchmark_all_backends(dataset_paths):
    results = run_vad_benchmark_all(
        split="val",
        dataset_root=dataset_paths.root,
        limit=2,
    )

    assert len(results) == len(available_backends())
    assert all(result.num_calls == 2 for result in results)


def test_vad_benchmark_missing_dataset_returns_empty():
    result = run_vad_benchmark(
        backend="energy",
        split="val",
        dataset_root="/nonexistent/path",
    )

    assert result.num_calls == 0
    assert result.caller.mean_iou == 0.0
    assert result.agent.mean_iou == 0.0
