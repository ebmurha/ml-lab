from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=1_600, freq="h", tz="UTC")
    hour = pd.Series(range(len(index)), index=index)
    return pd.DataFrame(
        {
            "pm25": 15 + (hour % 24) * 0.4 + hour * 0.01,
            "temperature": 20 + (hour % 24) * 0.1,
            "relativehumidity": 70 - (hour % 24) * 0.2,
        },
        index=index,
    )


@pytest.fixture
def raw_csv(tmp_path: Path, hourly_frame: pd.DataFrame) -> Path:
    units = {
        "pm25": "µg/m³",
        "temperature": "c",
        "relativehumidity": "%",
    }
    rows = []
    for sensor_id, signal in enumerate(units, start=1):
        for timestamp, value in hourly_frame[signal].items():
            rows.append(
                {
                    "sensor_id": sensor_id,
                    "parameter": signal,
                    "units": units[signal],
                    "value": value,
                    "datetime_utc_start": timestamp.isoformat(),
                    "has_flags": False,
                }
            )
    path = tmp_path / "measurements.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
