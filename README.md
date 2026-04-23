# Customer Churn Prediction

Predict which telecom customers are likely to churn and surface the strongest churn drivers — built as a portfolio-grade ML project with reproducible pipelines, documented evaluation, and an interactive dashboard.

## Project structure

```
src/
  data/          Load and validate raw CSVs against a schema contract
  features/      Feature engineering and train/test splitting
  models/        Baseline training, model persistence
  evaluation/    Classification metrics, SHAP explainability
  utils/         Path helpers
  run_train.py   CLI entry-point: ingest → features → train → evaluate → save
app/
  dashboard.py   Streamlit dashboard (metrics, SHAP plot, single-customer prediction)
tests/           Pytest suite (schema, preprocessing, metrics, imports)
docs/            Architecture, evaluation protocol, metrics dictionary, status
data/raw/        Immutable input CSVs (not committed)
reports/         Generated artifacts (metrics JSON, SHAP plots)
artifacts/       Saved models
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml,explain,viz]"
pre-commit install
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[all]"
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample, 7,043 customers, 21 columns, 26.54% churn rate). Download the CSV and place it at:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Models

Two models were trained and evaluated on the same stratified 80/20 split (seed=42, test set: 1,409 rows):

| Metric    | Logistic Regression | XGBoost |
|-----------|---------------------|---------|
| Accuracy  | 0.8070              | 0.7630  |
| Precision | 0.6584              | 0.5394  |
| Recall    | 0.5668              | 0.7326  |
| F1        | 0.6092              | 0.6213  |
| ROC-AUC   | 0.8418              | 0.8343  |
| Log-loss  | 0.4207              | 0.4828  |

**Logistic Regression is the default deployed model** (`artifacts/model.pkl`). It ranks better by ROC-AUC and is better calibrated, making it the safer choice when the cost of false positives relative to false negatives is unknown. **XGBoost is available as a recall-oriented alternative** (`artifacts/challenger_xgb.pkl`): it catches ~73% of churners versus ~57% for logistic regression, but at the cost of significantly more false positives (precision 0.54 vs 0.66). Choose XGBoost when missing a churner is materially more expensive than incorrectly flagging a retained customer.

## Train and evaluate

```bash
PYTHONPATH=src python -m run_train data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

This loads the raw CSV, validates the schema, builds features, trains both models on the same stratified split, prints a side-by-side comparison table to stdout, and saves:

- `artifacts/model.pkl` — Logistic Regression (default)
- `artifacts/challenger_xgb.pkl` — XGBoost (recall-oriented alternative)
- `reports/baseline_metrics.json` — Logistic Regression metrics
- `reports/challenger_metrics.json` — XGBoost metrics
- `reports/model_comparison.json` — side-by-side delta summary

## Dashboard

```bash
streamlit run app/dashboard.py
```

Shows baseline metrics, SHAP feature importance, and a form for single-customer churn prediction.

## Tests

```bash
pytest
```

27 tests covering schema validation, feature engineering, evaluation metrics, and import smoke checks.

## Quality gates

```bash
ruff check src tests
black --check src tests
mypy src
pytest
```

All four pass in CI (`.github/workflows/ci.yml`).

## Documentation

- `[docs/architecture.md](docs/architecture.md)` — dataset contract, raw schema, package layout
- `[docs/evaluation.md](docs/evaluation.md)` — split strategy, metrics, baseline vs challenger comparison, and business tradeoff analysis
- `[docs/metrics.md](docs/metrics.md)` — metric definitions and code locations
- `[docs/status.md](docs/status.md)` — milestone tracker
- `[TODO.md](TODO.md)` — backlog and next steps

## Status

See `[docs/status.md](docs/status.md)` for the current milestone and `[TODO.md](TODO.md)` for the task backlog.