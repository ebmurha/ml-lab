"""Evaluation helpers for ranking, calibration, and subgroup reporting."""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .config import FOLLOW_UP_CAPACITY


def probability_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def capacity_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    capacity: float = FOLLOW_UP_CAPACITY,
) -> dict[str, float | int]:
    count = max(1, math.ceil(len(y_true) * capacity))
    order = np.argsort(-np.asarray(probabilities), kind="stable")
    selected = np.zeros(len(y_true), dtype=bool)
    selected[order[:count]] = True
    truth = np.asarray(y_true, dtype=int)
    positives = int(truth.sum())
    captured = int(truth[selected].sum())
    return {
        "capacity_fraction": capacity,
        "selected_count": count,
        "readmissions": positives,
        "readmissions_captured": captured,
        "recall_at_capacity": float(captured / positives) if positives else 0.0,
    }


def subgroup_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    columns: tuple[str, ...] = ("race", "gender", "age"),
) -> dict[str, list[dict[str, object]]]:
    count = max(1, math.ceil(len(frame) * FOLLOW_UP_CAPACITY))
    selected = np.zeros(len(frame), dtype=bool)
    selected[np.argsort(-np.asarray(probabilities), kind="stable")[:count]] = True
    result: dict[str, list[dict[str, object]]] = {}
    for column in columns:
        values = frame[column].fillna("Missing").astype(str)
        rows = []
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            truth = frame.loc[mask, "target"].to_numpy(dtype=int)
            positives = int(truth.sum())
            true_positives = int(truth[selected[mask]].sum())
            recall = float(true_positives / positives) if positives else None
            rows.append(
                {
                    "group": value,
                    "sample_count": int(mask.sum()),
                    "positive_count": positives,
                    "recall": recall,
                    "false_negative_rate": None if recall is None else 1.0 - recall,
                }
            )
        result[column] = rows
    return result
