"""Model training entry-points and artifact persistence."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def train_challenger(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = 42,
) -> XGBClassifier:
    """Train an XGBoost challenger model on the same prepared features."""
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model


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
