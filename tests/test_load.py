"""Tests for the data loading layer."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data.load import load_raw_csv, load_telco_churn

FIXTURE = Path(__file__).parent / "fixtures" / "telco_sample.csv"


# --- load_raw_csv ---


def test_load_raw_csv_reads_tmp_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    got = load_raw_csv(csv_path)
    want = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    assert_frame_equal(got, want)


def test_load_raw_csv_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "missing.csv")


def test_load_raw_csv_empty_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="zero rows"):
        load_raw_csv(csv_path)


# --- load_telco_churn ---


def test_load_telco_churn_with_fixture() -> None:
    df = load_telco_churn(FIXTURE)
    assert len(df) > 0
    assert "Churn" in df.columns


def test_load_telco_churn_custom_target() -> None:
    df = load_telco_churn(FIXTURE, target_col="tenure")
    assert "tenure" in df.columns


def test_load_telco_churn_bad_target_raises() -> None:
    with pytest.raises(ValueError, match="Target column"):
        load_telco_churn(FIXTURE, target_col="NonExistent")


def test_load_telco_churn_skip_target_check() -> None:
    df = load_telco_churn(FIXTURE, target_col="")
    assert len(df) > 0
