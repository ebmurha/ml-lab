"""Load and validate OpenAQ measurements without temporal imputation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .config import SIGNAL_COLUMNS

REQUIRED_COLUMNS = {
    "parameter",
    "units",
    "value",
    "datetime_utc_start",
    "sensor_id",
    "has_flags",
}
ACCEPTED_UNITS = {
    "pm25": {"µg/m³", "ug/m3", "µg/m3"},
    "temperature": {"c", "°c"},
    "relativehumidity": {"%"},
}


@dataclass(frozen=True)
class DataProfile:
    source_rows: int
    selected_measurements: int
    hourly_rows: int
    start_utc: str
    end_utc: str
    missing_hours: int
    coverage_percent: float
    duplicate_measurements: int
    flagged_measurements: int
    missing_by_signal: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_hourly_measurements(path: str | Path) -> tuple[pd.DataFrame, DataProfile]:
    """Return one UTC-indexed hourly table and its aggregate quality profile."""
    source = pd.read_csv(path)
    source_row_count = len(source)
    missing_columns = REQUIRED_COLUMNS.difference(source.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    source["parameter"] = source["parameter"].str.lower()
    source = source[source["parameter"].isin(SIGNAL_COLUMNS)].copy()
    if source.empty:
        raise ValueError("dataset has none of the required forecasting signals")

    source["datetime_utc_start"] = pd.to_datetime(
        source["datetime_utc_start"], utc=True, errors="raise"
    )
    source["value"] = pd.to_numeric(source["value"], errors="coerce")
    flagged = source["has_flags"].astype(str).str.lower().eq("true")
    duplicate_count = int(
        source.duplicated(["sensor_id", "datetime_utc_start"], keep=False).sum()
    )
    _validate_units(source)

    usable = source.loc[~flagged & source["value"].notna()].copy()
    usable["hour"] = usable["datetime_utc_start"].dt.floor("h")
    hourly = (
        usable.groupby(["hour", "parameter"], observed=True)["value"]
        .mean()
        .unstack("parameter")
        .sort_index()
    )

    missing_signals = set(SIGNAL_COLUMNS).difference(hourly.columns)
    if missing_signals:
        raise ValueError(f"dataset is missing signals: {sorted(missing_signals)}")
    hourly = hourly.loc[:, list(SIGNAL_COLUMNS)]
    complete_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(complete_index)
    hourly.index.name = "timestamp_utc"

    observed_pm25 = int(hourly["pm25"].notna().sum())
    profile = DataProfile(
        source_rows=source_row_count,
        selected_measurements=len(source),
        hourly_rows=len(hourly),
        start_utc=hourly.index.min().isoformat(),
        end_utc=hourly.index.max().isoformat(),
        missing_hours=len(hourly) - observed_pm25,
        coverage_percent=round(100 * observed_pm25 / len(hourly), 2),
        duplicate_measurements=duplicate_count,
        flagged_measurements=int(flagged.sum()),
        missing_by_signal={
            column: int(hourly[column].isna().sum()) for column in SIGNAL_COLUMNS
        },
    )
    return hourly, profile


def _validate_units(source: pd.DataFrame) -> None:
    for parameter, expected in ACCEPTED_UNITS.items():
        actual = set(source.loc[source["parameter"].eq(parameter), "units"].dropna())
        normalized = {str(unit).lower() for unit in actual}
        if not normalized:
            continue
        if not normalized.issubset({unit.lower() for unit in expected}):
            raise ValueError(f"unexpected units for {parameter}: {sorted(actual)}")
