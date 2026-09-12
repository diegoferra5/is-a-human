import pytest

from is_a_human.eval.harness import run_foundation_eval


@pytest.mark.integration
def test_foundation_eval_on_val_subset(dataset_paths):
    summary = run_foundation_eval(split="val", dataset_root=dataset_paths.root, limit=5)

    assert summary.num_calls == 5
    assert summary.mean_caller_iou > 0.5
    assert summary.mean_agent_iou > 0.5
    assert summary.mean_latency_ms > 0
    assert summary.p95_latency_ms >= summary.mean_latency_ms * 0.5


def test_foundation_eval_missing_dataset_returns_empty():
    summary = run_foundation_eval(split="val", dataset_root="/nonexistent/path")

    assert summary.num_calls == 0
    assert summary.mean_caller_iou == 0.0
    assert summary.mean_agent_iou == 0.0
