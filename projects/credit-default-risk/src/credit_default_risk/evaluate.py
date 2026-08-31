"""Model evaluation and operational-threshold selection."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(
    y_true: Any, probabilities: np.ndarray, *, minimum_recall: float = 0.60
) -> float:
    """Select the highest-precision threshold satisfying the recall constraint."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= minimum_recall)
    if eligible.size == 0:
        return 0.5
    best_precision = precision[eligible].max()
    best = eligible[precision[eligible] == best_precision]
    return float(thresholds[best[-1]])


def classification_metrics(
    y_true: Any, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = probabilities >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
    }
