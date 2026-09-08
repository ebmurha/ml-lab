"""Leakage-safe features for six-hour-ahead PM2.5 forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZON_HOURS, LAG_HOURS, ROLLING_HOURS, SIGNAL_COLUMNS

TARGET_COLUMN = "target_pm25"
TARGET_TIME_COLUMN = "target_timestamp_utc"
PERSISTENCE_COLUMN = "persistence_prediction"
SEASONAL_COLUMN = "seasonal_naive_prediction"


def make_feature_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    """Build features using observations available at or before each row's time."""
    features: dict[str, pd.Series] = {}
    for signal in SIGNAL_COLUMNS:
        series = hourly[signal]
        features[f"{signal}_current"] = series
        for lag in LAG_HOURS:
            features[f"{signal}_lag_{lag}h"] = series.shift(lag)
        for window in ROLLING_HOURS:
            rolling = series.rolling(window, min_periods=window)
            features[f"{signal}_mean_{window}h"] = rolling.mean()
            features[f"{signal}_std_{window}h"] = rolling.std()

    horizon = pd.Timedelta(f"{HORIZON_HOURS}h")
    forecast_time = hourly.index + horizon
    hour_angle = 2 * np.pi * forecast_time.hour / 24
    weekday_angle = 2 * np.pi * forecast_time.dayofweek / 7
    features["forecast_hour_sin"] = pd.Series(np.sin(hour_angle), index=hourly.index)
    features["forecast_hour_cos"] = pd.Series(np.cos(hour_angle), index=hourly.index)
    features["forecast_weekday_sin"] = pd.Series(
        np.sin(weekday_angle), index=hourly.index
    )
    features["forecast_weekday_cos"] = pd.Series(
        np.cos(weekday_angle), index=hourly.index
    )
    return pd.DataFrame(features, index=hourly.index)


def make_supervised_dataset(hourly: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach the future target and two fair baseline predictions."""
    features = make_feature_frame(hourly)
    feature_columns = list(features.columns)
    dataset = features.copy()
    dataset[TARGET_COLUMN] = hourly["pm25"].shift(-HORIZON_HOURS)
    dataset[TARGET_TIME_COLUMN] = dataset.index + pd.Timedelta(f"{HORIZON_HOURS}h")
    dataset[PERSISTENCE_COLUMN] = hourly["pm25"]
    dataset[SEASONAL_COLUMN] = hourly["pm25"].shift(24 - HORIZON_HOURS)
    required = [TARGET_COLUMN, PERSISTENCE_COLUMN, SEASONAL_COLUMN]
    return dataset.dropna(subset=required), feature_columns
