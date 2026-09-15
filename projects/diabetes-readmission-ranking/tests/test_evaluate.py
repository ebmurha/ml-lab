import numpy as np
import pandas as pd

from diabetes_readmission_ranking.evaluate import capacity_metrics


def test_capacity_metrics_use_highest_scores():
    target = pd.Series([0, 1, 1, 0, 0])
    metrics = capacity_metrics(target, np.array([0.1, 0.9, 0.8, 0.2, 0.3]), capacity=0.4)
    assert metrics["selected_count"] == 2
    assert metrics["readmissions_captured"] == 2
    assert metrics["recall_at_capacity"] == 1.0
