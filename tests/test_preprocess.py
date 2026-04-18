"""Tests for the feature engineering pipeline."""

from pathlib import Path

import pandas as pd

from data.load import load_raw_csv
from features.preprocess import TARGET_COL, build_feature_matrix, split_data

FIXTURE = Path(__file__).parent / "fixtures" / "telco_sample.csv"


def test_build_feature_matrix_returns_numeric() -> None:
    raw = load_raw_csv(FIXTURE)
    result = build_feature_matrix(raw)
    non_numeric = result.select_dtypes(exclude="number").columns.tolist()
    assert non_numeric == [], f"Non-numeric columns remain: {non_numeric}"


def test_build_feature_matrix_has_target() -> None:
    raw = load_raw_csv(FIXTURE)
    result = build_feature_matrix(raw)
    assert TARGET_COL in result.columns


def test_build_feature_matrix_drops_customer_id() -> None:
    raw = load_raw_csv(FIXTURE)
    result = build_feature_matrix(raw)
    assert "customerID" not in result.columns


def test_total_charges_coerced() -> None:
    raw = load_raw_csv(FIXTURE)
    result = build_feature_matrix(raw)
    assert pd.api.types.is_float_dtype(result["TotalCharges"])
    assert result["TotalCharges"].isna().sum() == 0


def test_split_data_reproducible() -> None:
    raw = load_raw_csv(FIXTURE)
    features = build_feature_matrix(raw)
    X1, _, y1, _ = split_data(features, seed=0)
    X2, _, y2, _ = split_data(features, seed=0)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_split_data_no_target_leakage() -> None:
    raw = load_raw_csv(FIXTURE)
    features = build_feature_matrix(raw)
    X_train, X_test, _, _ = split_data(features)
    assert TARGET_COL not in X_train.columns
    assert TARGET_COL not in X_test.columns
