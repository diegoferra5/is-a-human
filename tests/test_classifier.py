import pytest

from is_a_human.analysis.classifier import evaluate_classifier, train_logistic_regression
from is_a_human.analysis.features import CallFeatures

pytestmark = pytest.mark.acoustic


def _row(label: str, caller_talk: float, agent_talk: float) -> CallFeatures:
    base = {name: 0.0 for name in CallFeatures.numeric_field_names()}
    base.update({"caller_talk_ratio": caller_talk, "agent_talk_ratio": agent_talk})
    return CallFeatures(anon_id=f"id_{label}_{caller_talk}", label=label, split="train", **base)


def test_logistic_classifier_separates_simple_classes(report_metrics):
    train_rows = [
        _row("human", 0.15, 0.55),
        _row("human", 0.18, 0.58),
        _row("human", 0.16, 0.57),
        _row("synthetic", 0.30, 0.40),
        _row("synthetic", 0.28, 0.42),
        _row("synthetic", 0.32, 0.38),
    ]
    model = train_logistic_regression(train_rows, ("caller_talk_ratio", "agent_talk_ratio"))
    metrics = evaluate_classifier(model, train_rows)
    report_metrics(accuracy=metrics.accuracy, auc=metrics.auc, f1=metrics.f1)

    assert metrics.accuracy >= 0.8
    assert metrics.auc >= 0.8
