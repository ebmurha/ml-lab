"""Tests for preprocessing and evaluation behavior."""

import numpy as np
import pandas as pd

from credit_default_risk.config import FEATURES
from credit_default_risk.evaluate import (
    classification_metrics,
    evaluation_diagnostics,
    select_threshold,
    subgroup_metrics,
)
from credit_default_risk.features import build_candidates


def test_threshold_meets_minimum_recall():
    truth = np.array([0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = select_threshold(truth, probabilities, minimum_recall=2 / 3)
    metrics = classification_metrics(truth, probabilities, threshold)
    assert metrics["recall"] >= 2 / 3
    assert 0 <= threshold <= 1


def test_candidate_pipelines_fit_without_leaking_preprocessing():
    rows = 80
    frame = pd.DataFrame(
        {feature: np.arange(rows) % 4 for feature in FEATURES}
    )
    frame["sex"] = 1 + np.arange(rows) % 2
    frame["education"] = 1 + np.arange(rows) % 4
    frame["marriage"] = 1 + np.arange(rows) % 3
    target = pd.Series(np.arange(rows) % 2)
    for candidate in build_candidates().values():
        candidate.fit(frame, target)
        probabilities = candidate.predict_proba(frame)[:, 1]
        assert probabilities.shape == (rows,)


def test_evaluation_diagnostics_are_aggregate_and_complete():
    truth = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    diagnostics = evaluation_diagnostics(truth, probabilities, threshold=0.5)
    assert diagnostics["confusion_matrix"].tolist() == [[2, 0], [0, 2]]
    assert len(diagnostics["false_positive_rate"]) == len(
        diagnostics["true_positive_rate"]
    )


def test_subgroup_metrics_do_not_expose_customer_rows():
    frame = pd.DataFrame({"sex": [1, 1, 2, 2], "education": [1, 2, 1, 2]})
    truth = np.array([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.7])
    result = subgroup_metrics(
        frame, truth, probabilities, threshold=0.5, columns=["sex"]
    )
    assert result["count"].tolist() == [2, 2]
    assert set(result.columns) == {
        "feature",
        "value",
        "count",
        "observed_default_rate",
        "predicted_high_risk_rate",
        "recall",
        "precision",
    }
