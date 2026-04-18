"""Model training entry-points and artifact persistence."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.paths import project_root

ARTIFACTS_DIR = project_root() / "artifacts"


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = 42,
) -> Pipeline:
    """Train a logistic regression baseline with standard scaling.

    Returns the fitted sklearn Pipeline.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model: Any, path: Path | None = None) -> Path:
    """Persist a fitted model to *path* using pickle. Returns the written path."""
    if path is None:
        path = ARTIFACTS_DIR / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: Path | None = None) -> Any:
    """Load a previously saved model."""
    if path is None:
        path = ARTIFACTS_DIR / "model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301
