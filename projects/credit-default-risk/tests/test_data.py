"""Tests for dataset loading and validation."""

import pandas as pd
import pytest

from credit_default_risk.config import FEATURES
from credit_default_risk.data import load_dataset, split_dataset


def _source_frame(rows: int = 100) -> pd.DataFrame:
    values = {feature.upper(): [index % 4 for index in range(rows)] for feature in FEATURES}
    values["ID"] = list(range(1, rows + 1))
    values["default payment next month"] = [index % 2 for index in range(rows)]
    values["SEX"] = [1 + index % 2 for index in range(rows)]
    values["EDUCATION"] = [0 if index == 0 else 2 for index in range(rows)]
    values["MARRIAGE"] = [0 if index == 0 else 1 for index in range(rows)]
    return pd.DataFrame(values)


def test_loader_drops_id_and_normalizes_undocumented_categories(monkeypatch):
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: _source_frame())
    frame = load_dataset("unused.xls")
    assert "id" not in frame
    assert frame.loc[0, "education"] == 4
    assert frame.loc[0, "marriage"] == 3


def test_loader_rejects_missing_required_column(monkeypatch):
    source = _source_frame().drop(columns=["AGE"])
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: source)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset("unused.xls")


def test_split_is_reproducible_stratified_60_20_20(monkeypatch):
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: _source_frame())
    split = split_dataset(load_dataset("unused.xls"))
    assert (len(split.X_train), len(split.X_validation), len(split.X_test)) == (60, 20, 20)
    assert split.y_train.mean() == split.y_validation.mean() == split.y_test.mean() == 0.5
