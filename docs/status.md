# Status

## Milestone overview (6)

| ID | Focus                             | State       |
| -- | --------------------------------- | ----------- |
| M1 | Data contract + ingest validation | Complete    |
| M2 | Feature pipeline + splits         | Complete    |
| M3 | Baseline train + artifacts        | Complete    |
| M4 | Evaluation + report outputs       | Complete    |
| M5 | README/docs sync + quality gates  | Complete    |
| M6 | Dataset ingestion + EDA           | Complete    |

## M0 — Repository scaffold (complete)

- Standard portfolio layout, Python 3.11+ tooling, pre-commit, and CI are in place.
- Starter library code: `utils.paths.project_root`, `data.load.load_raw_csv`.
- Smoke tests cover path resolution and CSV loading.

## M0b — Tooling and import surface (complete)

- `pyproject.toml` extended with optional dependency groups: `ml`, `explain`, `viz`, `app`, `notebooks`, `all`.
- Pydantic added to core dependencies for schema validation.
- Stub modules created and later replaced with real implementations.

## M1 — Data contract (complete)

- Dataset: Telco Customer Churn (IBM/Kaggle), documented in `docs/architecture.md`.
- Schema validation: `data.schema.validate_raw_dataframe`.
- Fixture: `tests/fixtures/telco_sample.csv` (6 rows including edge cases).
- Tests: `tests/test_schema.py` (5 tests).

## M2 — Feature pipeline (complete)

- `features.preprocess.build_feature_matrix`: binary encoding, TotalCharges coercion, one-hot encoding.
- `features.preprocess.split_data`: stratified 80/20 split, seed=42.
- Tests: `tests/test_preprocess.py` (6 tests covering numerics, target leakage, reproducibility).

## M3 — Baseline model (complete)

- `models.train.train_baseline`: LogisticRegression with StandardScaler pipeline.
- `models.train.save_model` / `load_model`: pickle-based persistence to `artifacts/`.
- CLI: `python -m run_train` from `src/`.

## M4 — Evaluation (complete)

- `evaluation.metrics.classification_report`: accuracy, precision, recall, F1, ROC-AUC, log-loss.
- `evaluation.explain.shap_summary`: SHAP beeswarm plot + top feature ranking.
- Artifacts: `reports/baseline_metrics.json`, `reports/shap_summary.png`.

## M5 — Polish (complete)

- README: quickstart, dataset, train, dashboard, tests, quality gates, docs links.
- Streamlit dashboard: `app/dashboard.py` with metrics, SHAP, and single-customer prediction.
- All quality gates passing: Ruff, Black, Mypy, Pytest.

## M6 — Dataset ingestion + EDA (complete)

- `data.load.load_telco_churn`: convenience loader with default path resolution (tries canonical filename then `churn.csv`), schema validation, empty-dataset guard, and configurable target column check.
- `data.load.load_raw_csv`: hardened with empty-dataset check (`ValueError` on zero rows).
- Real dataset placed at `data/raw/churn.csv` and notebook executed end-to-end.
- `notebooks/01_eda.ipynb` now contains executed outputs with real metrics and visuals (shape 7,043 x 21, churn Yes 26.54%, duplicates 0, hidden TotalCharges missing = 11 after coercion).
- Final observations section is completed with concise preprocessing-relevant insights.
- Tests: `tests/test_load.py` remains 7 tests (empty CSV, fixture loading, custom target, bad target, skip target check).

## Blockers / decisions

- None.
