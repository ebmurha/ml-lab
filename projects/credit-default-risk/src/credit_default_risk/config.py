"""Project configuration and feature schema."""

from pathlib import Path

RANDOM_SEED = 42
TARGET_COLUMN = "default"
MODEL_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "default-of-credit-card-clients.xls"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / f"model-{MODEL_VERSION}.joblib"

CATEGORICAL_FEATURES = ["sex", "education", "marriage"]
NUMERIC_FEATURES = [
    "limit_bal",
    "age",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "bill_amt1",
    "bill_amt2",
    "bill_amt3",
    "bill_amt4",
    "bill_amt5",
    "bill_amt6",
    "pay_amt1",
    "pay_amt2",
    "pay_amt3",
    "pay_amt4",
    "pay_amt5",
    "pay_amt6",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
