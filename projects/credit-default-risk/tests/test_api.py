"""Contract tests for the prediction API."""

import numpy as np
from fastapi.testclient import TestClient

from credit_default_risk.api import create_app
from credit_default_risk.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from credit_default_risk.inference import InferenceService


class FixedModel:
    def predict_proba(self, frame):
        probability = 0.7 if int(frame.iloc[0]["pay_0"]) >= 1 else 0.1
        return np.array([[1 - probability, probability]])


def _client() -> TestClient:
    bundle = {
        "model": FixedModel(),
        "model_version": "test-version",
        "threshold": 0.5,
        "medium_risk_threshold": 0.25,
        "features": FEATURES,
        "reference_profile": {
            "numeric": {
                feature: {"mean": 0.0, "std": 1.0} for feature in NUMERIC_FEATURES
            },
            "categories": {
                feature: [1, 2, 3, 4] for feature in CATEGORICAL_FEATURES
            },
        },
    }
    return TestClient(create_app(InferenceService(bundle)))


def _payload() -> dict[str, int]:
    return {
        "limit_bal": 20000,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 24,
        "pay_0": 2,
        "pay_2": 2,
        "pay_3": -1,
        "pay_4": -1,
        "pay_5": -2,
        "pay_6": -2,
        "bill_amt1": 3913,
        "bill_amt2": 3102,
        "bill_amt3": 689,
        "bill_amt4": 0,
        "bill_amt5": 0,
        "bill_amt6": 0,
        "pay_amt1": 0,
        "pay_amt2": 689,
        "pay_amt3": 0,
        "pay_amt4": 0,
        "pay_amt5": 0,
        "pay_amt6": 0,
    }


def test_health_returns_ok():
    assert _client().get("/health").json() == {"status": "ok"}


def test_predict_returns_probability():
    response = _client().post("/predict", json=_payload())
    assert response.status_code == 200
    assert response.json()["default_probability"] == 0.7


def test_predict_returns_high_risk_band():
    response = _client().post("/predict", json=_payload())
    assert response.json()["risk_band"] == "high"


def test_predict_returns_low_risk_band():
    payload = _payload()
    payload["pay_0"] = 0
    response = _client().post("/predict", json=payload)
    assert response.json()["risk_band"] == "low"


def test_predict_returns_model_version():
    response = _client().post("/predict", json=_payload())
    assert response.json()["model_version"] == "test-version"


def test_predict_rejects_missing_field():
    payload = _payload()
    del payload["age"]
    assert _client().post("/predict", json=payload).status_code == 422


def test_predict_rejects_extra_field():
    payload = {**_payload(), "id": 1}
    assert _client().post("/predict", json=payload).status_code == 422


def test_predict_rejects_unknown_sex_code():
    payload = {**_payload(), "sex": 3}
    assert _client().post("/predict", json=payload).status_code == 422


def test_predict_rejects_impossible_age():
    payload = {**_payload(), "age": 12}
    assert _client().post("/predict", json=payload).status_code == 422


def test_predict_rejects_invalid_payment_status():
    payload = {**_payload(), "pay_0": 10}
    assert _client().post("/predict", json=payload).status_code == 422
