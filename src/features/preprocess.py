"""Feature engineering: transform raw ingested data into a training matrix."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]

YES_NO_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

MULTI_CATEGORY_COLS = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]


def build_feature_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert validated raw data into model-ready features.

    Steps:
      1. Drop ``customerID`` (not predictive).
      2. Encode binary Yes/No columns as 0/1.
      3. Coerce ``TotalCharges`` to numeric (spaces become NaN, filled with 0).
      4. One-hot encode remaining categorical columns.

    Returns a DataFrame with a numeric ``Churn`` target column.
    """
    df = raw.copy()
    df = df.drop(columns=DROP_COLS, errors="ignore")

    for col in YES_NO_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    df = pd.get_dummies(df, columns=MULTI_CATEGORY_COLS, drop_first=True)

    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    return df


def split_data(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:  # type: ignore[type-arg]
    """Split feature matrix into train/test with reproducible seed.

    Returns (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    result: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return result
