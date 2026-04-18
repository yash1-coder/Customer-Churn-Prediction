# TODO

## Milestones (5)

### M1 — Data contract, label, horizon, and ingest validation

**Status:** Complete.

Dataset, label, horizon documented in `docs/architecture.md`. Schema validation in `data.schema` with `tests/test_schema.py`. Fixture at `tests/fixtures/telco_sample.csv`.

### M2 — Feature pipeline and reproducible splits

**Status:** Complete.

`features.preprocess.build_feature_matrix` encodes binary/categorical columns, coerces `TotalCharges`, and one-hot encodes multi-category features. `split_data` provides a stratified 80/20 split with fixed seed. Split strategy documented in `docs/evaluation.md`. Tests in `tests/test_preprocess.py`.

### M3 — Baseline model, training entrypoint, and artifacts

**Status:** Complete.

`models.train.train_baseline` fits a `LogisticRegression` pipeline with `StandardScaler`. CLI entry-point at `src/run_train.py`. Model saved to `artifacts/model.pkl`. Documented in README.

### M4 — Evaluation protocol and reportable outputs

**Status:** Complete.

`evaluation.metrics.classification_report` computes accuracy, precision, recall, F1, ROC-AUC, and log-loss. `evaluation.explain` generates SHAP summary plots. `run_train.py` writes `reports/baseline_metrics.json`. Metrics dictionary in `docs/metrics.md`.

### M5 — Portfolio polish and quality gates

**Status:** Complete.

README documents install, train, evaluate, and dashboard commands. All docs in sync. Ruff, Black, Mypy, and Pytest pass. Streamlit dashboard in `app/dashboard.py`.

## Done

- Scaffold folders, packaging, CI, and documentation skeleton.
- First code milestone: `project_root`, `load_raw_csv`, and unit tests.
- M0b: dependency groups, stub modules, import smoke tests.
- M1: data contract, schema validation, fixture data.
- M2: feature pipeline with reproducible splits.
- M3: baseline model, training CLI, artifact persistence.
- M4: evaluation metrics, SHAP explainability.
- M5: README, docs sync, quality gates, Streamlit dashboard.

### M6 — Dataset ingestion and EDA

**Status:** In progress (skeleton created).

`data.load.load_telco_churn` convenience loader with schema validation and fallback filenames. EDA notebook skeleton at `notebooks/01_eda.ipynb` with 8 sections (setup, load, missing values, duplicates, target balance, distributions, correlations, observations). Notebook needs to be run against the real dataset to fill in findings.

## Potential extensions

- XGBoost or gradient boosting challenger model.
- Hyperparameter tuning (Optuna or GridSearchCV).
- Threshold analysis and calibration.
- Docker containerisation.

## Hygiene

- `pre-commit install` documented in README.