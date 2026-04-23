"""Unit tests for src/api/utils.py — the refactored cleaning helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.api.utils import (
    clean_dataframe,
    clean_input_data_from_file,
    save_predictions,
    validate_data,
)


def test_clean_dataframe_drops_specified_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    out = clean_dataframe(df, drop_columns=["b"])
    assert list(out.columns) == ["a", "c"]


def test_clean_dataframe_silently_ignores_missing_drop_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = clean_dataframe(df, drop_columns=["never_existed"])
    assert list(out.columns) == ["a"]
    assert len(out) == 3


def test_clean_dataframe_drops_rows_with_nans_by_default():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
    out = clean_dataframe(df, handle_missing="drop")
    assert len(out) == 1
    assert out.iloc[0]["a"] == 1.0


def test_clean_dataframe_imputes_with_median():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0], "b": [2.0, 4.0, np.nan, 8.0]})
    out = clean_dataframe(df, handle_missing="impute", impute_strategy="median")
    # No NaNs remain, shape preserved.
    assert out.isna().sum().sum() == 0
    assert out.shape == df.shape


def test_clean_dataframe_does_not_mutate_input():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    before = df.copy()
    _ = clean_dataframe(df, drop_columns=["a"])
    pd.testing.assert_frame_equal(df, before)


def test_clean_dataframe_rejects_unknown_strategy():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="handle_missing"):
        clean_dataframe(df, handle_missing="nope")


def test_clean_input_data_from_file_roundtrips(tmp_path):
    src = tmp_path / "in.csv"
    dst = tmp_path / "sub" / "out.csv"
    pd.DataFrame({"a": [1, 2, 3], "drop_me": [9, 9, 9]}).to_csv(src, index=False)
    clean_input_data_from_file(str(src), str(dst), drop_columns=["drop_me"])
    written = pd.read_csv(dst)
    assert list(written.columns) == ["a"]


def test_validate_data_raises_on_missing_column():
    df = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_data(df, ["feature1", "feature2", "feature3"])


def test_validate_data_passes_when_all_present():
    df = pd.DataFrame({"feature1": [1.0], "feature2": [2.0], "feature3": [3.0]})
    validate_data(df, ["feature1", "feature2", "feature3"])  # no exception


def test_save_predictions_with_identifiers(tmp_path):
    out = tmp_path / "preds" / "preds.csv"
    save_predictions([1.1, 2.2], str(out), input_identifier=["a", "b"])
    written = pd.read_csv(out)
    assert list(written.columns) == ["Identifier", "Predicted_RUL"]
    assert len(written) == 2


def test_save_predictions_rejects_mismatched_identifiers(tmp_path):
    out = tmp_path / "preds.csv"
    with pytest.raises(ValueError):
        save_predictions([1.1, 2.2], str(out), input_identifier=["only-one"])
