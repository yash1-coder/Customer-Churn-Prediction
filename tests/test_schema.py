"""Tests for the raw data schema validator."""

from pathlib import Path

import pytest

from data.load import load_raw_csv
from data.schema import REQUIRED_COLUMNS, SchemaError, validate_raw_dataframe

FIXTURE = Path(__file__).parent / "fixtures" / "telco_sample.csv"


def test_valid_fixture_passes() -> None:
    df = load_raw_csv(FIXTURE)
    result = validate_raw_dataframe(df)
    assert set(result.columns) == REQUIRED_COLUMNS


def test_missing_column_raises() -> None:
    df = load_raw_csv(FIXTURE).drop(columns=["Churn"])
    with pytest.raises(SchemaError, match="Missing columns"):
        validate_raw_dataframe(df)


def test_wrong_dtype_raises() -> None:
    df = load_raw_csv(FIXTURE)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    with pytest.raises(SchemaError, match="Dtype mismatches"):
        validate_raw_dataframe(df)


def test_extra_columns_ignored() -> None:
    df = load_raw_csv(FIXTURE)
    df["extra_col"] = 1
    result = validate_raw_dataframe(df)
    assert "extra_col" in result.columns


def test_empty_dataframe_with_correct_schema() -> None:
    df = load_raw_csv(FIXTURE).iloc[:0]
    result = validate_raw_dataframe(df)
    assert len(result) == 0
