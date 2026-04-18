"""Raw data schema contract for the Telco Customer Churn dataset.

The authoritative column list and expected pandas dtypes live here so
that ingest code can fail fast on schema drift.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype

# Semantic type: "string", "integer", or "float".
EXPECTED_COLUMNS: dict[str, str] = {
    "customerID": "string",
    "gender": "string",
    "SeniorCitizen": "integer",
    "Partner": "string",
    "Dependents": "string",
    "tenure": "integer",
    "PhoneService": "string",
    "MultipleLines": "string",
    "InternetService": "string",
    "OnlineSecurity": "string",
    "OnlineBackup": "string",
    "DeviceProtection": "string",
    "TechSupport": "string",
    "StreamingTV": "string",
    "StreamingMovies": "string",
    "Contract": "string",
    "PaperlessBilling": "string",
    "PaymentMethod": "string",
    "MonthlyCharges": "float",
    "TotalCharges": "string",
    "Churn": "string",
}

REQUIRED_COLUMNS: frozenset[str] = frozenset(EXPECTED_COLUMNS)

_KIND_CHECKERS = {
    "string": is_string_dtype,
    "integer": is_integer_dtype,
    "float": is_float_dtype,
}


class SchemaError(Exception):
    """Raised when a DataFrame does not match the expected raw schema."""


def validate_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Check *df* against the Telco raw schema and return it unchanged.

    Raises
    ------
    SchemaError
        If required columns are missing or have unexpected dtypes.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaError(f"Missing columns: {sorted(missing)}")

    dtype_errors: list[str] = []
    for col, kind in EXPECTED_COLUMNS.items():
        checker = _KIND_CHECKERS[kind]
        if not checker(df[col]):
            dtype_errors.append(f"{col}: expected {kind}, got {df[col].dtype}")

    if dtype_errors:
        raise SchemaError("Dtype mismatches:\n  " + "\n  ".join(dtype_errors))

    return df
