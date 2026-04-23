# TODO

## Milestones (6)

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
- M6: dataset ingestion and EDA completed with real data execution and finalized observations.

### M6 — Dataset ingestion and EDA

**Status:** Complete.

Real dataset loaded from `data/raw/churn.csv` and notebook executed. `notebooks/01_eda.ipynb` now includes completed EDA outputs and final observations: shape 7,043 x 21, churn Yes 26.54% (1,869), churn No 73.46% (5,174), duplicates 0, and 11 hidden missing values in `TotalCharges` after numeric coercion. Preprocessing-relevant implications are documented.

## Potential extensions

- XGBoost or gradient boosting challenger model.
- Hyperparameter tuning (Optuna or GridSearchCV).
- Threshold analysis and calibration.
- Docker containerisation.

## Hygiene

- `pre-commit install` documented in README.