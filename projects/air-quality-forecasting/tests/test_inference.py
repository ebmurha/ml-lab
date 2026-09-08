from pathlib import Path

import joblib
import pandas as pd

from air_quality_forecasting.inference import build_forecast


def test_persistence_inference_returns_valid_record(
    raw_csv: Path, hourly_frame, tmp_path: Path
) -> None:
    artifact = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model_name": "persistence",
            "model_version": "test-version",
            "model": None,
            "feature_columns": [],
        },
        artifact,
    )
    prediction_time = hourly_frame.index[-10]

    result = build_forecast(raw_csv, artifact, prediction_time)

    assert result["forecast_time_utc"] == (
        prediction_time + pd.Timedelta("6h")
    ).isoformat()
    assert result["predicted_pm25_ug_m3"] == round(
        float(hourly_frame.at[prediction_time, "pm25"]), 4
    )
    assert result["absolute_error"] >= 0
