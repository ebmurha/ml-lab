"""Versioned-model loading, inference, and drift indicators."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class InferenceService:
    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle

    @classmethod
    def load(cls, artifact_path: str | Path) -> "InferenceService":
        return cls(joblib.load(artifact_path))

    def predict(self, values: dict[str, int | float]) -> dict[str, Any]:
        frame = pd.DataFrame([values], columns=self.bundle["features"])
        probability = float(self.bundle["model"].predict_proba(frame)[0, 1])
        high_threshold = float(self.bundle["threshold"])
        medium_threshold = float(self.bundle["medium_risk_threshold"])
        if probability >= high_threshold:
            risk_band = "high"
        elif probability >= medium_threshold:
            risk_band = "medium"
        else:
            risk_band = "low"
        return {
            "default_probability": probability,
            "risk_band": risk_band,
            "model_version": self.bundle["model_version"],
            "drift": self._drift_indicators(values),
        }

    def _drift_indicators(self, values: dict[str, int | float]) -> dict[str, Any]:
        profile = self.bundle["reference_profile"]
        standardized_differences = []
        for column, reference in profile["numeric"].items():
            standard_deviation = reference["std"]
            if standard_deviation > 0:
                standardized_differences.append(
                    abs(float(values[column]) - reference["mean"]) / standard_deviation
                )
        unseen_categories = [
            column
            for column, known in profile["categories"].items()
            if int(values[column]) not in known
        ]
        return {
            "max_numeric_standard_deviations": max(standardized_differences, default=0.0),
            "numeric_outlier_count": sum(value > 3 for value in standardized_differences),
            "unseen_categories": unseen_categories,
        }
