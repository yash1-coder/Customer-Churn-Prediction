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

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample, 7043 customers, 21 columns). Download the CSV and place it at:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Train and evaluate

```bash
python -m run_train
```

This loads the raw CSV, validates the schema, builds features, trains a logistic regression baseline, evaluates on a held-out 20% test set, and saves:

- `artifacts/model.pkl` — fitted sklearn pipeline
- `reports/baseline_metrics.json` — accuracy, precision, recall, F1, ROC-AUC, log-loss

Run from the `src/` directory or with `PYTHONPATH=src`.

## Dashboard

```bash
streamlit run app/dashboard.py
```

Shows baseline metrics, SHAP feature importance, and a form for single-customer churn prediction.

## Tests

```bash
pytest
```

22 tests covering schema validation, feature engineering, evaluation metrics, and import smoke checks.

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
- `[docs/evaluation.md](docs/evaluation.md)` — split strategy, metrics, baseline, explainability
- `[docs/metrics.md](docs/metrics.md)` — metric definitions and code locations
- `[docs/status.md](docs/status.md)` — milestone tracker
- `[TODO.md](TODO.md)` — backlog and next steps

## Status

See `[docs/status.md](docs/status.md)` for the current milestone and `[TODO.md](TODO.md)` for the task backlog.