"""Model construction and probability calibration."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_SEED
from .features import FeatureSchema, prepare_with_schema


MODEL_NAMES = ("dummy", "logistic_regression", "catboost")


def build_model(name: str, schema: FeatureSchema):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        numeric = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
            ]
        )
        transform = ColumnTransformer(
            [("numeric", numeric, list(schema.numeric)), ("categorical", categorical, list(schema.categorical))]
        )
        return Pipeline(
            [
                ("transform", transform),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=500,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=300,
            depth=7,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="PRAUC",
            random_seed=RANDOM_SEED,
            thread_count=4,
            verbose=False,
            allow_writing_files=False,
            cat_features=list(schema.categorical),
        )
    raise ValueError(f"Unknown model: {name}")


@dataclass
class ProbabilityCalibrator:
    method: str
    estimator: object

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        if self.method == "sigmoid":
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            return self.estimator.predict_proba(logits)[:, 1]
        return np.asarray(self.estimator.predict(clipped), dtype=float)


def fit_calibrators(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, ProbabilityCalibrator]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    sigmoid = LogisticRegression(random_state=RANDOM_SEED).fit(logits, y_true)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(clipped, y_true)
    return {
        "sigmoid": ProbabilityCalibrator("sigmoid", sigmoid),
        "isotonic": ProbabilityCalibrator("isotonic", isotonic),
    }


@dataclass
class ModelBundle:
    model_name: str
    estimator: object
    calibrator: ProbabilityCalibrator
    schema: FeatureSchema
    model_version: str

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        features = prepare_with_schema(frame, self.schema)
        raw = self.estimator.predict_proba(features)[:, 1]
        return self.calibrator.transform(raw)
