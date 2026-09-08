"""Dataset loading, validation, and reproducible splitting."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURES, RANDOM_SEED, TARGET_COLUMN

SOURCE_TARGET = "default payment next month"


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the UCI workbook and return normalized model-ready columns."""
    frame = pd.read_excel(path, sheet_name="Data", header=1)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.rename(columns={SOURCE_TARGET: TARGET_COLUMN})

    required = {"id", TARGET_COLUMN, *FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    if frame[list(required)].isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1")
    if frame["id"].duplicated().any():
        raise ValueError("Dataset IDs must be unique")

    frame["education"] = frame["education"].replace({0: 4, 5: 4, 6: 4})
    frame["marriage"] = frame["marriage"].replace({0: 3})
    return frame[[*FEATURES, TARGET_COLUMN]].copy()


def split_dataset(frame: pd.DataFrame) -> DatasetSplit:
    """Create stratified 60/20/20 train, validation, and test partitions."""
    X = frame[FEATURES]
    y = frame[TARGET_COLUMN]
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=y_holdout,
    )
    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )
