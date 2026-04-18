"""CLI entry-point: load, featurise, train, evaluate, and save."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from data.load import load_raw_csv
from data.schema import validate_raw_dataframe
from evaluation.metrics import classification_report
from features.preprocess import build_feature_matrix, split_data
from models.train import save_model, train_baseline
from utils.paths import project_root


def main(data_path: Path | None = None) -> None:
    if data_path is None:
        data_path = (
            project_root() / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )

    print(f"Loading data from {data_path}")
    raw = load_raw_csv(data_path)
    raw = validate_raw_dataframe(raw)
    print(f"  rows={len(raw)}, columns={len(raw.columns)}")

    print("Building feature matrix...")
    features = build_feature_matrix(raw)
    X_train, X_test, y_train, y_test = split_data(features)
    print(f"  train={len(X_train)}, test={len(X_test)}")

    print("Training baseline (LogisticRegression)...")
    model = train_baseline(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = classification_report(y_test.to_numpy(), y_pred, y_prob)
    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    model_path = save_model(model)
    print(f"\nModel saved to {model_path}")

    reports_dir = project_root() / "reports"
    reports_dir.mkdir(exist_ok=True)
    metrics_path = reports_dir / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(path)
