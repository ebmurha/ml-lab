"""Train and evaluate the air-quality forecasting candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import psutil

from .config import HORIZON_HOURS, HOLDOUT_DAYS, MODEL_PARAMETERS
from .data import load_hourly_measurements
from .evaluate import (
    BASELINES,
    MODEL_NAME,
    new_model,
    run_experiment,
    split_final_holdout,
)
from .features import TARGET_COLUMN, make_supervised_dataset


class PeakMemoryMonitor:
    def __init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> "PeakMemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, psutil.Process().memory_info().rss)

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
            self._stop.wait(0.05)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def train_project(
    data_path: Path,
    artifact_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    with PeakMemoryMonitor() as memory:
        hourly, profile = load_hourly_measurements(data_path)
        dataset, feature_columns = make_supervised_dataset(hourly)
        result = run_experiment(dataset, feature_columns)

        development, holdout, _ = split_final_holdout(dataset)
        reproducible = True
        if result.selected_model == MODEL_NAME:
            replica = new_model().fit(
                development[feature_columns], development[TARGET_COLUMN]
            )
            reproducible = bool(
                np.allclose(
                    replica.predict(holdout[feature_columns]),
                    result.holdout_predictions[MODEL_NAME],
                    rtol=0,
                    atol=1e-12,
                )
            )

        digest = file_digest(data_path)
        model_version = f"{result.selected_model}-{digest[:12]}"
        artifact = {
            "model_version": model_version,
            "model_name": result.selected_model,
            "model": result.model,
            "feature_columns": feature_columns,
            "horizon_hours": HORIZON_HOURS,
            "data_sha256": digest,
            "data_end_utc": profile.end_utc,
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, artifact_path)

        hgb_windows = result.validation_metrics.pivot(
            index="window", columns="candidate", values="mae"
        )
        beats_both_windows = int(
            (
                (hgb_windows[MODEL_NAME] < hgb_windows["persistence"])
                & (hgb_windows[MODEL_NAME] < hgb_windows["seasonal_naive_24h"])
            ).sum()
        )
        holdout = result.holdout_metrics.set_index("candidate")["mae"]
        metrics: dict[str, Any] = {
            "model_version": model_version,
            "selection_criterion": (
                "Lowest mean MAE across four expanding validation windows; "
                "ties are resolved by candidate name."
            ),
            "selected_model": result.selected_model,
            "data_profile": profile.to_dict(),
            "supervised_rows": len(dataset),
            "development_rows": result.development_rows,
            "holdout_rows": result.holdout_rows,
            "holdout_days": HOLDOUT_DAYS,
            "holdout_start_utc": result.holdout_start.isoformat(),
            "validation": frame_records(result.validation_metrics),
            "validation_summary": frame_records(result.validation_summary),
            "holdout": frame_records(result.holdout_metrics),
            "verification": {
                "hgb_beats_both_baselines_in_most_windows": beats_both_windows
                > len(hgb_windows) / 2,
                "hgb_windows_beating_both": beats_both_windows,
                "hgb_beats_both_baselines_on_holdout": bool(
                    holdout[MODEL_NAME]
                    < min(holdout[name] for name in BASELINES)
                ),
                "repeated_training_reproduces_predictions": reproducible,
            },
            "model_parameters": MODEL_PARAMETERS,
        }

    metrics["peak_memory_mb"] = round(memory.peak_bytes / 1024**2, 2)
    metrics["verification"]["peak_memory_below_6gb"] = (
        memory.peak_bytes < 6 * 1024**3
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--artifact", type=Path, default=Path("artifacts/forecast_model.joblib")
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("artifacts/metrics.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    metrics = train_project(args.data, args.artifact, args.metrics)
    holdout = {row["candidate"]: row["mae"] for row in metrics["holdout"]}
    print(f"Selected: {metrics['selected_model']}")
    print(f"Holdout MAE: {holdout[metrics['selected_model']]:.3f}")
    print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")
    print(f"Completed in {time.perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()
