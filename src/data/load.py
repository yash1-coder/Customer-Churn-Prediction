from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.schema import validate_raw_dataframe
from utils.paths import project_root

_RAW_DIR = project_root() / "data" / "raw"
_CANONICAL_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
_ALT_NAME = "churn.csv"


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load a CSV from ``data/raw`` (or any path) without mutating the source file."""
    if not path.is_file():
        msg = f"Expected a CSV file at: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)


def load_telco_churn(path: Path | None = None) -> pd.DataFrame:
    """Load and validate the Telco Customer Churn dataset.

    Parameters
    ----------
    path:
        Explicit CSV path.  When *None* the function looks for
        ``WA_Fn-UseC_-Telco-Customer-Churn.csv`` then ``churn.csv``
        inside ``data/raw/``.

    Returns
    -------
    pd.DataFrame
        Validated raw DataFrame (schema checked by
        ``data.schema.validate_raw_dataframe``).

    Raises
    ------
    FileNotFoundError
        If no CSV can be found at the resolved path.
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
    return validate_raw_dataframe(df)
