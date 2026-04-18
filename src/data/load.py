from pathlib import Path

import pandas as pd


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load a CSV from ``data/raw`` (or any path) without mutating the source file."""
    if not path.is_file():
        msg = f"Expected a CSV file at: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)
