"""Model evaluation and operational-threshold selection."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
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


def evaluation_diagnostics(
    y_true: Any, probabilities: np.ndarray, threshold: float
) -> dict[str, np.ndarray]:
    """Return aggregate curve and confusion-matrix data for reporting."""
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    observed_rate, predicted_rate = calibration_curve(
        y_true, probabilities, n_bins=10, strategy="quantile"
    )
    return {
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "observed_rate": observed_rate,
        "predicted_rate": predicted_rate,
        "confusion_matrix": confusion_matrix(y_true, probabilities >= threshold),
    }


def subgroup_metrics(
    frame: pd.DataFrame,
    y_true: Any,
    probabilities: np.ndarray,
    threshold: float,
    columns: list[str],
) -> pd.DataFrame:
    """Summarize aggregate outcomes and errors for named subgroups."""
    truth = np.asarray(y_true)
    predictions = probabilities >= threshold
    rows: list[dict[str, Any]] = []
    for column in columns:
        for value in sorted(frame[column].unique()):
            mask = frame[column].to_numpy() == value
            rows.append(
                {
                    "feature": column,
                    "value": int(value),
                    "count": int(mask.sum()),
                    "observed_default_rate": float(truth[mask].mean()),
                    "predicted_high_risk_rate": float(predictions[mask].mean()),
                    "recall": float(
                        recall_score(truth[mask], predictions[mask], zero_division=0)
                    ),
                    "precision": float(
                        precision_score(truth[mask], predictions[mask], zero_division=0)
                    ),
                }
            )
    return pd.DataFrame(rows)
