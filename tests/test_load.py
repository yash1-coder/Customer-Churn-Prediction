from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data.load import load_raw_csv


def test_load_raw_csv_reads_tmp_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    got = load_raw_csv(csv_path)
    want = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    assert_frame_equal(got, want)


def test_load_raw_csv_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "missing.csv")
