# Metrics

Metrics computed by `evaluation.metrics.classification_report`.

| Metric | Definition | Code location | Notes |
| --- | --- | --- | --- |
| Accuracy | Fraction of correct predictions | `evaluation.metrics` | Baseline sanity check; not primary due to class imbalance |
| Precision | TP / (TP + FP) | `evaluation.metrics` | Cost of false positives (unnecessary retention spend) |
| Recall | TP / (TP + FN) | `evaluation.metrics` | Cost of missed churners — primary metric |
| F1 | Harmonic mean of precision and recall | `evaluation.metrics` | Single summary of the precision-recall trade-off |
| ROC-AUC | Area under the ROC curve | `evaluation.metrics` | Requires predicted probabilities |
| Log-loss | Negative log-likelihood of predictions | `evaluation.metrics` | Requires predicted probabilities; measures calibration |
