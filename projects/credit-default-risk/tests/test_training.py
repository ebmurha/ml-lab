"""Tests for preprocessing and evaluation behavior."""

import numpy as np
import pandas as pd

from credit_default_risk.config import FEATURES
from credit_default_risk.evaluate import classification_metrics, select_threshold
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
