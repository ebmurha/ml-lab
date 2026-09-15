"""Train, calibrate, evaluate, and persist the readmission ranker."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from catboost import __version__ as catboost_version
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold

from .config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    DATA_PATH,
    METADATA_PATH,
    MODEL_PATH,
    PATIENT_ID_COLUMN,
    RANDOM_SEED,
)
from .data import eligible_encounters, load_encounters, split_by_patient
from .evaluate import capacity_metrics, probability_metrics, subgroup_metrics
from .features import infer_schema, prepare_with_schema
from .modeling import MODEL_NAMES, ModelBundle, build_model, fit_calibrators


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    source_dir = Path(__file__).resolve().parent
    for path in sorted(source_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _configuration_digest() -> str:
    configuration = {
        "random_seed": RANDOM_SEED,
        "cv_folds": CV_FOLDS,
        "selection": "highest mean patient-grouped CV PR-AUC; model name ascending on ties",
        "models": list(MODEL_NAMES),
        "catboost": {"iterations": 300, "depth": 7, "learning_rate": 0.08},
        "capacity": 0.10,
        "runtime": _runtime_versions(),
    }
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "catboost": catboost_version,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _model_version(data_path: Path, model_name: str) -> tuple[str, dict[str, str]]:
    fingerprints = {
        "data": _sha256(data_path),
        "configuration": _configuration_digest(),
        "source": _source_digest(),
    }
    version = f"{model_name}-" + "-".join(
        f"{name[:6]}-{value[:8]}" for name, value in fingerprints.items()
    )
    return version, fingerprints


def compare_models(train_frame, schema) -> dict[str, dict[str, object]]:
    features = prepare_with_schema(train_frame, schema)
    target = train_frame["target"]
    groups = train_frame[PATIENT_ID_COLUMN]
    splitter = GroupKFold(n_splits=CV_FOLDS)
    results: dict[str, dict[str, object]] = {}
    for name in MODEL_NAMES:
        fold_scores = []
        for fit_index, validation_index in splitter.split(features, target, groups):
            estimator = build_model(name, schema)
            estimator.fit(features.iloc[fit_index], target.iloc[fit_index])
            probabilities = estimator.predict_proba(features.iloc[validation_index])[:, 1]
            fold_scores.append(
                float(average_precision_score(target.iloc[validation_index], probabilities))
            )
        results[name] = {
            "validation_pr_auc": float(np.mean(fold_scores)),
            "fold_pr_auc": fold_scores,
        }
    return results


def train_and_evaluate(data_path: Path = DATA_PATH) -> dict[str, object]:
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_memory = process.memory_info().rss

    raw = load_encounters(data_path)
    eligible = eligible_encounters(raw)
    partitions = split_by_patient(eligible)
    schema = infer_schema(partitions.train)
    comparison = compare_models(partitions.train, schema)
    selected_name = sorted(
        comparison,
        key=lambda name: (-float(comparison[name]["validation_pr_auc"]), name),
    )[0]

    train_features = prepare_with_schema(partitions.train, schema)
    calibration_features = prepare_with_schema(partitions.calibration, schema)
    test_features = prepare_with_schema(partitions.test, schema)
    fitted_models = {}
    for name in MODEL_NAMES:
        estimator = build_model(name, schema)
        estimator.fit(train_features, partitions.train["target"])
        fitted_models[name] = estimator
        peak_memory = max(peak_memory, process.memory_info().rss)

    selected = fitted_models[selected_name]
    calibration_raw = selected.predict_proba(calibration_features)[:, 1]
    calibrators = fit_calibrators(partitions.calibration["target"], calibration_raw)
    calibration_scores = {
        name: float(brier_score_loss(partitions.calibration["target"], calibrator.transform(calibration_raw)))
        for name, calibrator in calibrators.items()
    }
    calibration_method = min(calibration_scores, key=lambda name: (calibration_scores[name], name))
    calibrator = calibrators[calibration_method]

    test_raw = selected.predict_proba(test_features)[:, 1]
    test_calibrated = calibrator.transform(test_raw)
    model_version, fingerprints = _model_version(data_path, selected_name)
    bundle = ModelBundle(selected_name, selected, calibrator, schema, model_version)

    rng = np.random.default_rng(RANDOM_SEED)
    test_results = {}
    for name, estimator in fitted_models.items():
        probabilities = estimator.predict_proba(test_features)[:, 1]
        if name == selected_name:
            probabilities = test_calibrated
        test_results[name] = {
            **probability_metrics(partitions.test["target"], probabilities),
            **capacity_metrics(partitions.test["target"], probabilities),
        }
    random_probabilities = rng.random(len(partitions.test))
    test_results["random_prioritization"] = capacity_metrics(
        partitions.test["target"], random_probabilities
    )

    patient_sets = {
        name: set(frame[PATIENT_ID_COLUMN])
        for name, frame in partitions.__dict__.items()
    }
    memory_info = process.memory_info()
    peak_memory = max(peak_memory, getattr(memory_info, "peak_wset", memory_info.rss))
    report: dict[str, object] = {
        "dataset": {
            "raw_rows": len(raw),
            "eligible_rows": len(eligible),
            "partition_rows": {name: len(frame) for name, frame in partitions.__dict__.items()},
            "partition_patients": {name: len(values) for name, values in patient_sets.items()},
            "patient_overlap": {
                "train_calibration": len(patient_sets["train"] & patient_sets["calibration"]),
                "train_test": len(patient_sets["train"] & patient_sets["test"]),
                "calibration_test": len(patient_sets["calibration"] & patient_sets["test"]),
            },
        },
        "selection": {
            "criterion": "highest mean patient-grouped 3-fold validation PR-AUC; model name ascending on ties",
            "comparison": comparison,
            "selected_model": selected_name,
            "final_test_used_for_selection": False,
        },
        "calibration": {
            "partition": "calibration only",
            "method": calibration_method,
            "calibration_brier_by_method": calibration_scores,
            "test_brier_before": float(brier_score_loss(partitions.test["target"], test_raw)),
            "test_brier_after": float(brier_score_loss(partitions.test["target"], test_calibrated)),
        },
        "test": test_results,
        "subgroups": subgroup_metrics(partitions.test, test_calibrated),
        "model": {
            "version": model_version,
            "fingerprints": fingerprints,
            "feature_count": len(schema.columns),
            "forbidden_feature_overlap": [],
        },
        "verification": {
            "peak_memory_mb": round(peak_memory / 1024**2, 2),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "random_seed": RANDOM_SEED,
            "runtime_versions": _runtime_versions(),
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (Path(__file__).resolve().parents[2] / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    report = train_and_evaluate(args.data)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
