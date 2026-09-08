"""Chronological model comparison and final holdout evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .config import HOLDOUT_DAYS, MODEL_PARAMETERS, VALIDATION_WINDOWS
from .features import (
    PERSISTENCE_COLUMN,
    SEASONAL_COLUMN,
    TARGET_COLUMN,
    TARGET_TIME_COLUMN,
)

MODEL_NAME = "hist_gradient_boosting"
BASELINES = {
    "persistence": PERSISTENCE_COLUMN,
    "seasonal_naive_24h": SEASONAL_COLUMN,
}


@dataclass
class ExperimentResult:
    selected_model: str
    model: HistGradientBoostingRegressor | None
    feature_columns: list[str]
    holdout_start: pd.Timestamp
    validation_metrics: pd.DataFrame
    validation_summary: pd.DataFrame
    holdout_metrics: pd.DataFrame
    holdout_predictions: pd.DataFrame
    development_rows: int
    holdout_rows: int


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**MODEL_PARAMETERS)


def split_final_holdout(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    holdout_start = dataset.index.max() - pd.Timedelta(f"{HOLDOUT_DAYS}d")
    holdout = dataset.loc[dataset.index >= holdout_start].copy()
    development = dataset.loc[dataset[TARGET_TIME_COLUMN] < holdout_start].copy()
    if development.empty or holdout.empty:
        raise ValueError("insufficient data for the configured chronological holdout")
    return development, holdout, holdout_start


def expanding_windows(
    development: pd.DataFrame,
    windows: int = VALIDATION_WINDOWS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    initial_size = int(len(development) * 0.6)
    validation_positions = np.array_split(
        np.arange(initial_size, len(development)), windows
    )
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for positions in validation_positions:
        if len(positions) == 0:
            continue
        validation = development.iloc[positions]
        validation_start = validation.index.min()
        training = development.loc[
            development[TARGET_TIME_COLUMN] < validation_start
        ]
        if training.empty:
            continue
        splits.append((training, validation))
    if len(splits) != windows:
        raise ValueError("insufficient data for expanding-window validation")
    return splits


def run_experiment(
    dataset: pd.DataFrame, feature_columns: list[str]
) -> ExperimentResult:
    development, holdout, holdout_start = split_final_holdout(dataset)
    validation_rows: list[dict[str, object]] = []

    for window, (training, validation) in enumerate(
        expanding_windows(development), start=1
    ):
        model = new_model().fit(training[feature_columns], training[TARGET_COLUMN])
        predictions = {
            MODEL_NAME: model.predict(validation[feature_columns]),
            **{
                name: validation[column].to_numpy()
                for name, column in BASELINES.items()
            },
        }
        for candidate, predicted in predictions.items():
            validation_rows.append(
                {
                    "window": window,
                    "candidate": candidate,
                    "mae": mean_absolute_error(validation[TARGET_COLUMN], predicted),
                    "training_rows": len(training),
                    "validation_rows": len(validation),
                    "validation_start": validation.index.min(),
                    "validation_end": validation.index.max(),
                }
            )

    validation_metrics = pd.DataFrame(validation_rows)
    validation_summary = validation_metrics.groupby(
        "candidate", as_index=False
    ).agg(mean_mae=("mae", "mean"))
    winners = validation_metrics.loc[
        validation_metrics.groupby("window")["mae"].idxmin(), "candidate"
    ].value_counts()
    validation_summary["windows_won"] = (
        validation_summary["candidate"].map(winners).fillna(0).astype(int)
    )
    validation_summary = validation_summary.sort_values(
        ["mean_mae", "candidate"], ignore_index=True
    )
    selected = str(validation_summary.iloc[0]["candidate"])

    fitted_model = new_model().fit(
        development[feature_columns], development[TARGET_COLUMN]
    )
    holdout_predictions = pd.DataFrame(
        {
            "actual_pm25": holdout[TARGET_COLUMN],
            MODEL_NAME: fitted_model.predict(holdout[feature_columns]),
            **{name: holdout[column] for name, column in BASELINES.items()},
        },
        index=holdout.index,
    )
    holdout_metrics = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "mae": mean_absolute_error(
                    holdout_predictions["actual_pm25"],
                    holdout_predictions[candidate],
                ),
            }
            for candidate in (MODEL_NAME, *BASELINES)
        ]
    ).sort_values("mae", ignore_index=True)

    return ExperimentResult(
        selected_model=selected,
        model=fitted_model if selected == MODEL_NAME else None,
        feature_columns=feature_columns,
        holdout_start=holdout_start,
        validation_metrics=validation_metrics,
        validation_summary=validation_summary,
        holdout_metrics=holdout_metrics,
        holdout_predictions=holdout_predictions,
        development_rows=len(development),
        holdout_rows=len(holdout),
    )
