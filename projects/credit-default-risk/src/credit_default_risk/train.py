"""Train, calibrate, evaluate, and persist the selected model."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .config import (
    CATEGORICAL_FEATURES,
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_DATA_PATH,
    MODEL_VERSION,
    NUMERIC_FEATURES,
)
from .data import load_dataset, split_dataset
from .evaluate import classification_metrics, select_threshold
from .features import build_candidates


def _reference_profile(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = {
        column: {
            "mean": float(frame[column].mean()),
            "std": float(frame[column].std(ddof=0)),
        }
        for column in NUMERIC_FEATURES
    }
    categories = {
        column: sorted(int(value) for value in frame[column].unique())
        for column in CATEGORICAL_FEATURES
    }
    return {"numeric": numeric, "categories": categories}


def train_model(data_path: str | Path, artifact_path: str | Path) -> dict[str, Any]:
    frame = load_dataset(data_path)
    split = split_dataset(frame)
    candidate_results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}

    for name, candidate in build_candidates().items():
        calibrated = CalibratedClassifierCV(candidate, method="sigmoid", cv=5, n_jobs=1)
        calibrated.fit(split.X_train, split.y_train)
        validation_probability = calibrated.predict_proba(split.X_validation)[:, 1]
        candidate_results[name] = classification_metrics(
            split.y_validation, validation_probability, threshold=0.5
        )
        fitted[name] = calibrated

    selected_name = max(
        candidate_results,
        key=lambda name: (
            candidate_results[name]["roc_auc"],
            -candidate_results[name]["brier_score"],
        ),
    )
    selected_model = fitted[selected_name]
    validation_probability = selected_model.predict_proba(split.X_validation)[:, 1]
    threshold = select_threshold(split.y_validation, validation_probability)
    validation_metrics = classification_metrics(
        split.y_validation, validation_probability, threshold
    )
    test_probability = selected_model.predict_proba(split.X_test)[:, 1]
    test_metrics = classification_metrics(split.y_test, test_probability, threshold)

    bundle = {
        "model": selected_model,
        "model_version": MODEL_VERSION,
        "selected_model": selected_name,
        "threshold": threshold,
        "medium_risk_threshold": threshold / 2,
        "features": list(split.X_train.columns),
        "reference_profile": _reference_profile(split.X_train),
        "candidate_validation_metrics": candidate_results,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_rows": int(len(split.X_train)),
        "validation_rows": int(len(split.X_validation)),
        "test_rows": int(len(split.X_test)),
        "positive_rate": float(np.mean(frame["default"])),
    }

    destination = Path(artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return bundle


def _report(bundle: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key != "model"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args()
    bundle = train_model(args.data, args.output)
    print(json.dumps(_report(bundle), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
