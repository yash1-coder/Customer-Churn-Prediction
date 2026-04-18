# Architecture

## Design intent

Keep a thin boundary between **immutable raw inputs** (`data/raw`), **reproducible transforms** (`src/`), and **presentation or reporting** (`reports/`, optional `app/`).

## Dataset

**Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset, publicly available on Kaggle).

**Churn label:** The `Churn` column (`Yes` / `No`). A customer is considered churned if they left the service within the observation period captured by the dataset. The binary target is encoded as `1` (churned) / `0` (retained).

**Prediction horizon:** The dataset is a snapshot — each row represents a customer's state at observation time with a retrospective churn flag. The model predicts whether a customer *will churn* given their current account and service profile. There is no explicit time horizon to tune; the implicit horizon is "within the provider's churn window" as defined by IBM's data generation.

**Raw file:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (7043 rows, 21 columns).

### Raw schema contract

| Column | Dtype | Notes |
| --- | --- | --- |
| customerID | object | Unique identifier, dropped before modelling |
| gender | object | Male / Female |
| SeniorCitizen | int64 | 0 / 1 |
| Partner | object | Yes / No |
| Dependents | object | Yes / No |
| tenure | int64 | Months with provider |
| PhoneService | object | Yes / No |
| MultipleLines | object | Yes / No / No phone service |
| InternetService | object | DSL / Fiber optic / No |
| OnlineSecurity | object | Yes / No / No internet service |
| OnlineBackup | object | Yes / No / No internet service |
| DeviceProtection | object | Yes / No / No internet service |
| TechSupport | object | Yes / No / No internet service |
| StreamingTV | object | Yes / No / No internet service |
| StreamingMovies | object | Yes / No / No internet service |
| Contract | object | Month-to-month / One year / Two year |
| PaperlessBilling | object | Yes / No |
| PaymentMethod | object | Four payment types |
| MonthlyCharges | float64 | |
| TotalCharges | string | Contains spaces for zero-tenure rows; must be coerced to float during feature engineering |
| Churn | string | Yes / No — the target |

Validation in `data.schema` rejects DataFrames missing required columns or having incompatible dtype kinds (string/integer/float). See `data.schema.EXPECTED_COLUMNS` for the authoritative mapping. The checker uses `pandas.api.types` helpers so it works across both legacy `object` and the newer `StringDtype` backends.

## Python packages

Packages live directly under `src/` (`data`, `features`, `models`, `evaluation`, `utils`) so imports match the folder names you see in the tree.

Trade-off: the `data` package name is generic. If this ever collides with tooling or habits (`import data` ambiguity), rename the package to a project-specific namespace (for example `churn_data`) and update imports once.

## Building blocks

- `utils.paths.project_root` — locate the repository root without hard-coded paths.
- `data.load.load_raw_csv` — read-only CSV ingest for raw files.
- `data.schema` — Pydantic-based raw schema definition and `validate_raw_dataframe` guard.
- `features.preprocess` — training matrix construction (stub, M2).
- `models.train` — baseline trainer and model persistence (stub, M3).
- `evaluation.metrics` — classification metrics aligned with `docs/metrics.md` (stub, M4).
