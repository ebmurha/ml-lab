"""Dataset loading, eligibility rules, and patient-level partitioning."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CALIBRATION_FRACTION,
    DATA_PATH,
    INELIGIBLE_DISCHARGE_DISPOSITIONS,
    PATIENT_ID_COLUMN,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_FRACTION,
    TRAIN_FRACTION,
)


@dataclass(frozen=True)
class DataPartitions:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def load_encounters(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the UCI encounter file without modifying its raw representation."""
    frame = pd.read_csv(path, na_values=["?"], low_memory=False)
    required = {"encounter_id", PATIENT_ID_COLUMN, TARGET_COLUMN, "discharge_disposition_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def eligible_encounters(frame: pd.DataFrame) -> pd.DataFrame:
    """Return discharges eligible for follow-up and create a binary target."""
    eligible = frame.loc[
        ~frame["discharge_disposition_id"].isin(INELIGIBLE_DISCHARGE_DISPOSITIONS)
        & frame[PATIENT_ID_COLUMN].notna()
        & frame[TARGET_COLUMN].isin(["NO", ">30", "<30"])
    ].copy()
    eligible = eligible.drop_duplicates(subset="encounter_id", keep="first")
    eligible["target"] = (eligible[TARGET_COLUMN] == "<30").astype("int8")
    return eligible.reset_index(drop=True)


def _patient_labels(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby(PATIENT_ID_COLUMN, sort=False)["target"].max()


def split_by_patient(
    frame: pd.DataFrame,
    *,
    random_seed: int = RANDOM_SEED,
) -> DataPartitions:
    """Create reproducible partitions in which each patient appears exactly once."""
    if abs(TRAIN_FRACTION + CALIBRATION_FRACTION + TEST_FRACTION - 1.0) > 1e-9:
        raise ValueError("Partition fractions must sum to one")

    labels = _patient_labels(frame)
    train_ids, remainder_ids = train_test_split(
        labels.index.to_numpy(),
        test_size=1.0 - TRAIN_FRACTION,
        random_state=random_seed,
        stratify=labels.to_numpy(),
    )
    remainder_labels = labels.loc[remainder_ids]
    calibration_ids, test_ids = train_test_split(
        remainder_ids,
        test_size=TEST_FRACTION / (CALIBRATION_FRACTION + TEST_FRACTION),
        random_state=random_seed,
        stratify=remainder_labels.to_numpy(),
    )

    partitions = DataPartitions(
        train=frame.loc[frame[PATIENT_ID_COLUMN].isin(train_ids)].reset_index(drop=True),
        calibration=frame.loc[frame[PATIENT_ID_COLUMN].isin(calibration_ids)].reset_index(drop=True),
        test=frame.loc[frame[PATIENT_ID_COLUMN].isin(test_ids)].reset_index(drop=True),
    )
    assert_patient_separation(partitions)
    return partitions


def assert_patient_separation(partitions: DataPartitions) -> None:
    groups = [set(part[PATIENT_ID_COLUMN]) for part in partitions.__dict__.values()]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Patient overlap detected between partitions")
