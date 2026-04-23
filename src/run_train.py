"""CLI entry-point: load, featurise, train, evaluate, and save."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

from data.load import load_raw_csv
from data.schema import validate_raw_dataframe
from evaluation.metrics import classification_report
from features.preprocess import build_feature_matrix, split_data
from models.train import save_model, train_baseline, train_challenger
from utils.paths import project_root


def _save_json(data: Mapping[str, float] | dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _print_comparison_table(
    baseline_metrics: Mapping[str, float], challenger_metrics: Mapping[str, float]
) -> None:
    metric_names = list(baseline_metrics.keys())
    print("\nModel comparison (same split):")
    print(f"{'metric':<12} {'baseline_lr':>12} {'challenger_xgb':>16} {'delta':>10}")
    print("-" * 54)
    for metric in metric_names:
        baseline_value = baseline_metrics[metric]
        challenger_value = challenger_metrics[metric]
        delta = challenger_value - baseline_value
        row = (
            f"{metric:<12} {baseline_value:>12.4f} "
            f"{challenger_value:>16.4f} {delta:>+10.4f}"
        )
        print(row)


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
    baseline_model = train_baseline(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_prob = baseline_model.predict_proba(X_test)[:, 1]
    baseline_metrics = classification_report(
        y_test.to_numpy(), baseline_pred, baseline_prob
    )

    print("\nBaseline evaluation metrics:")
    for name, value in baseline_metrics.items():
        print(f"  {name}: {value:.4f}")

    baseline_model_path = save_model(baseline_model)
    print(f"\nBaseline model saved to {baseline_model_path}")

    print("\nTraining challenger (XGBClassifier)...")
    challenger_model = train_challenger(X_train, y_train)
    challenger_pred = challenger_model.predict(X_test)
    challenger_prob = challenger_model.predict_proba(X_test)[:, 1]
    challenger_metrics = classification_report(
        y_test.to_numpy(), challenger_pred, challenger_prob
    )

    reports_dir = project_root() / "reports"
    reports_dir.mkdir(exist_ok=True)
    baseline_metrics_path = reports_dir / "baseline_metrics.json"
    _save_json(baseline_metrics, baseline_metrics_path)
    print(f"Baseline metrics saved to {baseline_metrics_path}")

    challenger_model_path = save_model(
        challenger_model, project_root() / "artifacts" / "challenger_xgb.pkl"
    )
    print(f"Challenger model saved to {challenger_model_path}")

    challenger_metrics_path = reports_dir / "challenger_metrics.json"
    _save_json(challenger_metrics, challenger_metrics_path)
    print(f"Challenger metrics saved to {challenger_metrics_path}")

    delta_metrics = {
        metric: challenger_metrics[metric] - baseline_metrics[metric]
        for metric in baseline_metrics
    }
    comparison_payload: dict[str, object] = {
        "baseline": dict(baseline_metrics),
        "challenger_xgb": dict(challenger_metrics),
        "delta_challenger_minus_baseline": delta_metrics,
    }
    comparison_path = reports_dir / "model_comparison.json"
    _save_json(comparison_payload, comparison_path)
    print(f"Model comparison saved to {comparison_path}")

    _print_comparison_table(baseline_metrics, challenger_metrics)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(path)
