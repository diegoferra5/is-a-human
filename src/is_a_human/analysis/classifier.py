"""Simple logistic baseline for separability estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from is_a_human.analysis.features import CallFeatures, numeric_feature_vector


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class ClassifierMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    features_used: tuple[str, ...]


@dataclass(frozen=True)
class TrainedLogistic:
    feature_names: tuple[str, ...]
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray

    def predict_proba_rows(self, rows: list[CallFeatures]) -> np.ndarray:
        matrix = np.vstack([self._transform(row) for row in rows])
        return _sigmoid(matrix @ self.weights + self.bias)

    def predict_rows(self, rows: list[CallFeatures], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba_rows(rows) >= threshold).astype(int)

    def predict_one(self, row) -> float:
        return float(self.predict_proba_rows([row])[0])

    def _transform(self, row) -> np.ndarray:
        values = np.array([float(getattr(row, name)) for name in self.feature_names])
        return (values - self.mean) / self.std

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainedLogistic":
        return cls(
            feature_names=tuple(payload["feature_names"]),
            weights=np.array(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            mean=np.array(payload["mean"], dtype=np.float64),
            std=np.array(payload["std"], dtype=np.float64),
        )


def _binary_labels(rows: list[CallFeatures]) -> np.ndarray:
    return np.array([1.0 if row.label == "synthetic" else 0.0 for row in rows])


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> ClassifierMetrics:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    order = np.argsort(y_prob)
    sorted_true = y_true[order]
    n_pos = np.sum(sorted_true == 1)
    n_neg = np.sum(sorted_true == 0)
    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        ranks = np.arange(1, len(sorted_true) + 1)
        pos_rank_sum = float(np.sum(ranks[sorted_true == 1]))
        auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return ClassifierMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        features_used=tuple(),
    )


def train_logistic_regression(
    train_rows: list[CallFeatures],
    feature_names: tuple[str, ...],
    *,
    learning_rate: float = 0.05,
    epochs: int = 2500,
    l2: float = 0.01,
    sample_weight: np.ndarray | None = None,
) -> TrainedLogistic:
    """sample_weight: optional per-row weights (normalised to mean 1). Lets a head
    be trained with extra emphasis on the rows where it is actually used."""
    matrix = np.vstack(
        [np.array([float(getattr(row, name)) for name in feature_names]) for row in train_rows]
    )
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    matrix = (matrix - mean) / std

    labels = _binary_labels(train_rows)
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    bias = 0.0
    sw = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    sw = sw * (len(sw) / sw.sum())

    for _ in range(epochs):
        logits = matrix @ weights + bias
        probs = _sigmoid(logits)
        error = (probs - labels) * sw
        grad_w = (matrix.T @ error) / len(labels) + l2 * weights
        grad_b = float(np.mean(error))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return TrainedLogistic(
        feature_names=feature_names,
        weights=weights,
        bias=bias,
        mean=mean,
        std=std,
    )


def evaluate_classifier(
    model: TrainedLogistic,
    rows: list[CallFeatures],
) -> ClassifierMetrics:
    labels = _binary_labels(rows)
    probs = model.predict_proba_rows(rows)
    preds = (probs >= 0.5).astype(int)
    metrics = _compute_metrics(labels, preds, probs)
    return ClassifierMetrics(
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        auc=metrics.auc,
        features_used=model.feature_names,
    )
