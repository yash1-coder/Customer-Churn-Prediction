# Evaluation

## Dataset and target

**Dataset:** Telco Customer Churn (IBM/Kaggle, 7043 rows, 21 columns).

**Churn label:** `Churn` column — binary Yes/No, encoded to 1/0. See `docs/architecture.md` for the full schema.

**Prediction horizon:** Snapshot-based; the model predicts whether a customer will churn given their current account profile.

## Split strategy

Stratified 80/20 train/test split with `random_state=42` via `features.preprocess.split_data`. Stratification ensures the churn class ratio is preserved in both sets.

## Metrics

See `docs/metrics.md` for the full table. Primary metric is **recall** (minimise missed churners). Secondary metrics: F1, ROC-AUC, precision.

## Baseline

Logistic regression with `StandardScaler`, trained via `models.train.train_baseline`. Serves as a performance floor for future models (XGBoost, etc.).

## Explainability

SHAP values via `evaluation.explain.shap_summary` — produces a beeswarm plot under `reports/shap_summary.png` and returns a ranking of top features by mean |SHAP value|.
