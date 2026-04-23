"""Shared helpers for the prediction API.

Previously all cleaning went through a disk round-trip — the API handler
wrote the request to a CSV, ran :func:`clean_input_data_from_file`, read
the cleaned CSV back, and deleted both temp files. That pattern doesn't
survive concurrent traffic (two requests race on the same ``temp_*.csv``
paths) and it's an order of magnitude slower than it needs to be for
what is ultimately a pandas operation.

:func:`clean_dataframe` is the in-memory path every live request should
use. The legacy file-based wrapper stays for scripts and tests that
already call it.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
from sklearn.impute import SimpleImputer


def clean_dataframe(
    df: pd.DataFrame,
    *,
    drop_columns: Optional[List[str]] = None,
    handle_missing: str = "drop",
    impute_strategy: str = "mean",
) -> pd.DataFrame:
    """Clean an in-memory DataFrame: drop columns, handle NaNs.

    Returns a new DataFrame; the caller's frame is not mutated.

    Args:
        df: Input DataFrame.
        drop_columns: Column names to drop if present. Missing ones are
            silently ignored (matches the prior behaviour of
            ``clean_input_data_from_file``).
        handle_missing: ``"drop"`` to discard rows with NaN, ``"impute"``
            to fill via :class:`~sklearn.impute.SimpleImputer`.
        impute_strategy: Strategy passed to ``SimpleImputer`` when
            ``handle_missing="impute"``.

    Raises:
        ValueError: If ``handle_missing`` is not ``"drop"`` or ``"impute"``.
    """
    out = df.copy()

    if drop_columns:
        to_drop = [c for c in drop_columns if c in out.columns]
        if to_drop:
            out = out.drop(columns=to_drop)

    if handle_missing == "drop":
        out = out.dropna().reset_index(drop=True)
    elif handle_missing == "impute":
        if out.empty:
            return out
        imputer = SimpleImputer(strategy=impute_strategy)
        imputed = imputer.fit_transform(out)
        out = pd.DataFrame(imputed, columns=out.columns)
    else:
        raise ValueError("handle_missing must be either 'drop' or 'impute'.")

    return out


def clean_input_data_from_file(
    input_path: str,
    output_path: str,
    drop_columns: Optional[List[str]] = None,
    handle_missing: str = "drop",
    impute_strategy: str = "mean",
) -> None:
    """File-to-file wrapper around :func:`clean_dataframe`.

    Kept for scripts and tests that already wrote CSVs; new API code
    should prefer the in-memory entry point.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input data file not found at {input_path}")

    df = pd.read_csv(input_path)
    cleaned = clean_dataframe(
        df,
        drop_columns=drop_columns,
        handle_missing=handle_missing,
        impute_strategy=impute_strategy,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cleaned.to_csv(output_path, index=False)


# Back-compat alias — older imports may still reference this name.
clean_input_data = clean_input_data_from_file


def save_predictions(
    predictions: List[float],
    output_path: str,
    input_identifier: Optional[List[str]] = None,
) -> None:
    """Save a list of predictions to a CSV, optionally with identifiers."""
    if input_identifier is not None:
        if len(input_identifier) != len(predictions):
            raise ValueError("Length of input_identifier must match length of predictions.")
        df = pd.DataFrame({"Identifier": input_identifier, "Predicted_RUL": predictions})
    else:
        df = pd.DataFrame({"Predicted_RUL": predictions})

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)


def validate_data(df: pd.DataFrame, required_features: List[str]) -> None:
    """Raise ``ValueError`` if any required feature column is missing."""
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
