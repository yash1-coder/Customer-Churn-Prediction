"""Data loading utilities for the Telco Customer Churn dataset.

Public API
----------
load_raw_csv
    Low-level CSV reader with file-existence check.
load_telco_churn
    High-level loader: resolves default path, validates schema,
    checks for empty data and target column presence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.schema import validate_raw_dataframe
from utils.paths import project_root

_RAW_DIR = project_root() / "data" / "raw"
_CANONICAL_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
_ALT_NAME = "churn.csv"

DEFAULT_TARGET = "Churn"


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    path:
        Absolute or relative path to a CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    ValueError
        If the loaded dataset has zero rows.
    """
    if not path.is_file():
        msg = f"Expected a CSV file at: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)

    if df.empty:
        msg = f"Dataset at {path} loaded but contains zero rows."
        raise ValueError(msg)

    return df


def load_telco_churn(
    path: Path | None = None,
    *,
    target_col: str = DEFAULT_TARGET,
) -> pd.DataFrame:
    """Load, validate, and sanity-check the Telco Customer Churn dataset.

    Parameters
    ----------
    path:
        Explicit CSV path.  When *None* the function looks for
        ``WA_Fn-UseC_-Telco-Customer-Churn.csv`` then ``churn.csv``
        inside ``data/raw/``.
    target_col:
        Name of the binary target column to verify.  Defaults to
        ``"Churn"``.  Set to ``""`` or ``None`` to skip the check.

    Returns
    -------
    pd.DataFrame
        Schema-validated raw DataFrame ready for exploration or
        feature engineering.

    Raises
    ------
    FileNotFoundError
        If no CSV can be found at the resolved path.
    ValueError
        If the dataset is empty or the target column is missing.
    data.schema.SchemaError
        If the loaded DataFrame fails schema validation.
    """
    if path is None:
        canonical = _RAW_DIR / _CANONICAL_NAME
        alt = _RAW_DIR / _ALT_NAME
        if canonical.is_file():
            path = canonical
        elif alt.is_file():
            path = alt
        else:
            msg = f"Telco CSV not found. Place the dataset at " f"{canonical} or {alt}"
            raise FileNotFoundError(msg)

    df = load_raw_csv(path)
    df = validate_raw_dataframe(df)

    if target_col and target_col not in df.columns:
        msg = f"Target column '{target_col}' not found. Columns: {list(df.columns)}"
        raise ValueError(msg)

    return df
