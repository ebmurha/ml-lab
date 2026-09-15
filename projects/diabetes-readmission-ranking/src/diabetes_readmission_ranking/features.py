"""Shared feature-schema and preparation logic."""

from dataclasses import dataclass

import pandas as pd

from .config import CATEGORICAL_CODE_COLUMNS, FORBIDDEN_FEATURES


@dataclass(frozen=True)
class FeatureSchema:
    columns: tuple[str, ...]
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    category_levels: dict[str, tuple[str, ...]]


def infer_schema(frame: pd.DataFrame) -> FeatureSchema:
    columns = tuple(column for column in frame.columns if column not in FORBIDDEN_FEATURES)
    forbidden = set(columns) & FORBIDDEN_FEATURES
    if forbidden:
        raise ValueError(f"Forbidden features selected: {sorted(forbidden)}")
    categorical = tuple(
        column
        for column in columns
        if frame[column].dtype == "object" or column in CATEGORICAL_CODE_COLUMNS
    )
    numeric = tuple(column for column in columns if column not in categorical)
    prepared = prepare_features(frame, columns, categorical, numeric)
    levels = {
        column: tuple(sorted(prepared[column].astype(str).unique()))
        for column in categorical
    }
    return FeatureSchema(columns, categorical, numeric, levels)


def prepare_features(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    categorical: tuple[str, ...],
    numeric: tuple[str, ...],
) -> pd.DataFrame:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    prepared = frame.loc[:, columns].copy()
    for column in categorical:
        prepared[column] = prepared[column].fillna("__MISSING__").astype(str)
    for column in numeric:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def prepare_with_schema(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    return prepare_features(frame, schema.columns, schema.categorical, schema.numeric)


def input_quality(frame: pd.DataFrame, schema: FeatureSchema) -> dict[str, object]:
    missing_columns = sorted(set(schema.columns).difference(frame.columns))
    if missing_columns:
        return {"missing_columns": missing_columns, "missing_values": {}, "unseen_categories": {}}
    prepared = prepare_with_schema(frame, schema)
    missing_values = {
        column: int(frame[column].isna().sum())
        for column in schema.columns
        if frame[column].isna().any()
    }
    unseen = {}
    for column in schema.categorical:
        count = int((~prepared[column].isin(schema.category_levels[column])).sum())
        if count:
            unseen[column] = count
    return {
        "missing_columns": [],
        "missing_values": missing_values,
        "unseen_categories": unseen,
    }
