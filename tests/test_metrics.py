"""Tests for evaluation metrics."""

import numpy as np

from evaluation.metrics import classification_report


def test_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    result = classification_report(y_true, y_pred)
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0


def test_with_probabilities() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    result = classification_report(y_true, y_pred, y_prob)
    assert "roc_auc" in result
    assert "log_loss" in result
    assert result["roc_auc"] == 1.0


def test_all_wrong_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    result = classification_report(y_true, y_pred)
    assert result["accuracy"] == 0.0
    assert result["recall"] == 0.0
