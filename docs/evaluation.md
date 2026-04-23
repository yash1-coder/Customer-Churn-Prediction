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

## Challenger: XGBoost

`XGBClassifier` with `scale_pos_weight = n_negative / n_positive` (computed from the training split) to compensate for class imbalance. No hyperparameter tuning beyond the default XGBoost settings. Trained via `models.train.train_challenger` on the same 80/20 stratified split (seed=42) used by the baseline.

## Model comparison

| Metric    | Logistic Regression | XGBoost | Delta   |
| --------- | ------------------- | ------- | ------- |
| Accuracy  | 0.8070              | 0.7630  | -0.0440 |
| Precision | 0.6584              | 0.5394  | -0.1190 |
| Recall    | 0.5668              | 0.7326  | +0.1658 |
| F1        | 0.6092              | 0.6213  | +0.0121 |
| ROC-AUC   | 0.8418              | 0.8343  | -0.0075 |
| Log-loss  | 0.4207              | 0.4828  | +0.0621 |

XGBoost is **not** a universal improvement. It gains recall at the direct expense of precision, accuracy, and calibration (log-loss). The ranking quality (ROC-AUC) is marginally worse, and the probability estimates are less reliable.

## Acceptance criteria result

M7 defined three acceptance gates for the challenger:

| Criterion      | Threshold | XGBoost | Result |
| -------------- | --------- | ------- | ------ |
| Recall         | >= 0.60   | 0.7326  | Pass   |
| F1             | >= 0.62   | 0.6213  | Pass   |
| ROC-AUC        | >= 0.85   | 0.8343  | Fail   |

**Verdict:** XGBoost passes 2 of 3 criteria. It fails the ROC-AUC gate (0.8343 vs 0.85 required). The baseline also falls short of the ROC-AUC threshold (0.8418 < 0.85), so neither model clears all three gates.

## Business tradeoff

The decision between the two models is a direct cost tradeoff:

- **XGBoost (high recall):** Catches ~73% of actual churners (vs ~57% for the baseline). In a 1,409-customer test set this means roughly 60 additional churners identified. The cost: for every correct churn flag, XGBoost also produces more false alarms — customers incorrectly flagged as at-risk. If the retention intervention is cheap (e.g., a targeted email or small discount), this tradeoff is worth it because a missed churner is more expensive than a wasted offer.

- **Logistic Regression (higher precision):** Flags fewer customers overall but is correct more often when it does flag (~66% precision vs ~54%). Better when the intervention is expensive (e.g., dedicated account management, large discount) and the business cannot afford to waste resources on false positives.

Neither model is "better" in the abstract. The right choice depends on the cost ratio of a missed churner versus a false alarm.

## Recommendation

**Retain Logistic Regression as the default deployed model.** Rationale:

1. It has better overall ranking (ROC-AUC 0.8418 vs 0.8343) and calibration (log-loss 0.4207 vs 0.4828), meaning its probability scores are more trustworthy for downstream use.
2. It has higher precision, which is the safer default when the business cost of interventions is unknown.
3. XGBoost's F1 improvement is marginal (+0.012) — not enough to justify worse calibration and precision without an explicit business mandate.

**Keep XGBoost available** (`artifacts/challenger_xgb.pkl`) as a recall-optimised alternative. If the business confirms that missing a churner is significantly more costly than a false alarm, switch to XGBoost or explore threshold tuning on the baseline's probability scores to achieve a better precision-recall operating point without changing models.

**Next steps to close the ROC-AUC gap (both models < 0.85):**
- Hyperparameter tuning (especially XGBoost: `max_depth`, `learning_rate`, `n_estimators`).
- Feature engineering: interaction terms, tenure buckets, contract-value ratios.
- Threshold tuning on the baseline to improve recall without switching models.
- Calibration analysis (Platt scaling or isotonic regression) if probability scores are used for prioritisation.

## Explainability

SHAP values via `evaluation.explain.shap_summary` — produces a beeswarm plot under `reports/shap_summary.png` and returns a ranking of top features by mean |SHAP value|.
