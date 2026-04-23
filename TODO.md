# TODO

## Milestones (7)

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

## Next

- **M7**: Implement `train_challenger()` in `src/models/train.py` (XGBoost with default hyperparams on existing feature matrix).

### M6 — Dataset ingestion and EDA

**Status:** Complete.

Real dataset loaded from `data/raw/churn.csv` and notebook executed. `notebooks/01_eda.ipynb` now includes completed EDA outputs and final observations: shape 7,043 x 21, churn Yes 26.54% (1,869), churn No 73.46% (5,174), duplicates 0, and 11 hidden missing values in `TotalCharges` after numeric coercion. Preprocessing-relevant implications are documented.

### M7 — XGBoost challenger model

**Status:** Planned.

**Scope:** Add `train_challenger()` to `models.train`, extend `run_train.py` to run both baseline and challenger on the same split, save challenger artifacts (`artifacts/challenger_xgb.pkl`, `reports/challenger_metrics.json`, `reports/model_comparison.json`), generate SHAP for XGBoost, update docs with comparison table.

**Acceptance criteria:**
- [ ] recall >= 0.60 (up from 0.5668 baseline)
- [ ] roc_auc >= 0.85 (at least matches baseline 0.8418)
- [ ] f1 >= 0.62 (recall gain not from precision collapse)
- [ ] `reports/model_comparison.json` exists with both models' metrics
- [ ] SHAP summary generated for XGBoost challenger
- [ ] `docs/status.md` and `docs/evaluation.md` updated with comparison table
- [ ] Ruff, Black, Mypy, Pytest pass

**Files to edit:** `src/models/train.py`, `src/run_train.py`, `docs/status.md`, `docs/evaluation.md`, `TODO.md`, `README.md`.

**Not in scope:** hyperparameter tuning, threshold analysis, dashboard changes, new features.

## Potential extensions

- Hyperparameter tuning (Optuna or GridSearchCV).
- Threshold analysis and calibration.
- Docker containerisation.

## Hygiene

- `pre-commit install` documented in README.