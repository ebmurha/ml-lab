from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from air_quality_forecasting.evaluate import MODEL_NAME
from air_quality_forecasting.features import TARGET_COLUMN, make_feature_frame
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


def test_selected_model_inference_returns_valid_record(
    raw_csv: Path, hourly_frame, tmp_path: Path
) -> None:
    features = make_feature_frame(hourly_frame)
    target = hourly_frame["pm25"].shift(-6).rename(TARGET_COLUMN)
    training = features.join(target).dropna(subset=[TARGET_COLUMN])
    model = HistGradientBoostingRegressor(max_iter=5, random_state=42).fit(
        training[features.columns], training[TARGET_COLUMN]
    )
    artifact = tmp_path / "selected-model.joblib"
    joblib.dump(
        {
            "model_name": MODEL_NAME,
            "model_version": "selected-model-test",
            "model": model,
            "feature_columns": list(features.columns),
        },
        artifact,
    )

    result = build_forecast(raw_csv, artifact, hourly_frame.index[-10])

    assert result["model_name"] == MODEL_NAME
    assert isinstance(result["predicted_pm25_ug_m3"], float)
    assert 0 <= result["missing_input_rate"] <= 1
