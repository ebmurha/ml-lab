"""Produce a timestamped PM2.5 forecast from downloaded OpenAQ data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd

from .config import HORIZON_HOURS
from .data import load_hourly_measurements
from .evaluate import MODEL_NAME
from .features import make_feature_frame


def build_forecast(
    data_path: Path,
    artifact_path: Path,
    prediction_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifact = joblib.load(artifact_path)
    hourly, _ = load_hourly_measurements(data_path)
    features = make_feature_frame(hourly)

    if prediction_time is None:
        candidates = hourly.index[hourly["pm25"].notna()]
        prediction_time = candidates.max()
    else:
        prediction_time = pd.Timestamp(prediction_time)
        if prediction_time.tzinfo is None:
            raise ValueError("prediction time must include a timezone")
        prediction_time = prediction_time.tz_convert("UTC")
    if prediction_time not in features.index:
        raise ValueError(f"prediction time is unavailable: {prediction_time}")

    model_name = artifact["model_name"]
    if model_name == MODEL_NAME:
        feature_row = features.loc[[prediction_time], artifact["feature_columns"]]
        predicted = float(artifact["model"].predict(feature_row)[0])
        missing_rate = float(feature_row.isna().mean(axis=1).iloc[0])
    elif model_name == "persistence":
        predicted = float(hourly.at[prediction_time, "pm25"])
        missing_rate = 0.0
    elif model_name == "seasonal_naive_24h":
        source_time = prediction_time - pd.Timedelta("18h")
        predicted = float(hourly.at[source_time, "pm25"])
        missing_rate = 0.0
    else:
        raise ValueError(f"unsupported model in artifact: {model_name}")

    forecast_time = prediction_time + pd.Timedelta(f"{HORIZON_HOURS}h")
    now = pd.Timestamp.now(tz="UTC")
    result: dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "prediction_time_utc": prediction_time.isoformat(),
        "forecast_time_utc": forecast_time.isoformat(),
        "horizon_hours": HORIZON_HOURS,
        "predicted_pm25_ug_m3": round(predicted, 4),
        "model_name": model_name,
        "model_version": artifact["model_version"],
        "data_freshness_hours": round(
            max(0.0, (now - prediction_time).total_seconds() / 3600), 2
        ),
        "missing_input_rate": round(missing_rate, 4),
    }
    if forecast_time in hourly.index and pd.notna(hourly.at[forecast_time, "pm25"]):
        actual = float(hourly.at[forecast_time, "pm25"])
        result["observed_pm25_ug_m3"] = actual
        result["absolute_error"] = round(abs(actual - predicted), 4)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def append_log(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def parse_timestamp(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("prediction time must include a timezone")
    return parsed.tz_convert("UTC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/forecast_model.joblib")
    )
    parser.add_argument("--prediction-time", type=parse_timestamp)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log", type=Path, default=Path("logs/forecasts.jsonl"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    record = build_forecast(args.data, args.artifact, args.prediction_time)
    append_log(record, args.log)
    rendered = json.dumps(record, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
