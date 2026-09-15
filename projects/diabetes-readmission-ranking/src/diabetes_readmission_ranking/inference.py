"""Rank discharge records for a fixed follow-up capacity."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import FOLLOW_UP_CAPACITY, MODEL_PATH
from .evaluate import probability_metrics
from .features import input_quality


def _emit(event: dict[str, object], log_path: Path | None) -> None:
    line = json.dumps(event, separators=(",", ":"))
    print(line)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def rank_discharges(
    frame: pd.DataFrame,
    bundle,
    *,
    capacity: float = FOLLOW_UP_CAPACITY,
    log_path: Path | None = None,
) -> pd.DataFrame:
    started = time.perf_counter()
    quality = input_quality(frame, bundle.schema)
    if quality["missing_columns"]:
        _emit(
            {
                "event": "schema_error",
                "timestamp": datetime.now(UTC).isoformat(),
                "model_version": bundle.model_version,
                **quality,
            },
            log_path,
        )
        raise ValueError(f"Input is missing required columns: {quality['missing_columns']}")

    scores = bundle.predict_proba(frame)
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(frame), dtype=int)
    ranks[order] = np.arange(1, len(frame) + 1)
    selected_count = max(1, math.ceil(len(frame) * capacity))

    result = pd.DataFrame(index=frame.index)
    for identifier in ("encounter_id", "patient_nbr"):
        if identifier in frame:
            result[identifier] = frame[identifier]
    result["readmission_probability"] = scores
    result["rank"] = ranks
    result["selected_for_follow_up"] = ranks <= selected_count
    result = result.sort_values("rank", kind="stable").reset_index(drop=True)

    event: dict[str, object] = {
        "event": "batch_scored",
        "timestamp": datetime.now(UTC).isoformat(),
        "model_version": bundle.model_version,
        "row_count": len(frame),
        "capacity_fraction": capacity,
        "selected_count": selected_count,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "score_distribution": {
            "minimum": float(np.min(scores)),
            "median": float(np.median(scores)),
            "maximum": float(np.max(scores)),
            "mean": float(np.mean(scores)),
        },
        **quality,
    }
    if "target" in frame or "readmitted" in frame:
        target = (
            frame["target"].astype(int)
            if "target" in frame
            else frame["readmitted"].eq("<30").astype(int)
        )
        event["delayed_outcome_metrics"] = probability_metrics(target, scores)
    _emit(event, log_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--capacity", type=float, default=FOLLOW_UP_CAPACITY)
    args = parser.parse_args()
    if not 0 < args.capacity <= 1:
        parser.error("--capacity must be greater than 0 and no more than 1")
    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.input, na_values=["?"], low_memory=False)
    ranked = rank_discharges(frame, bundle, capacity=args.capacity, log_path=args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
