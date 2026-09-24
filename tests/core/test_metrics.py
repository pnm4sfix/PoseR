"""Tests for poser.core.metrics."""

from __future__ import annotations

import logging

import numpy as np

from poser.core.metrics import benchmark_model_performance

RETURNED_KEYS = {
    "accuracy",
    "balanced_accuracy",
    "confusion_matrix",
    "classification_report",
}


def test_returns_expected_keys():
    targets = np.array([0, 1, 2, 0, 1, 2])
    result = benchmark_model_performance(targets, targets)
    assert set(result) == RETURNED_KEYS


def test_perfect_predictions_score_one():
    targets = np.array([0, 1, 2, 0, 1, 2])
    result = benchmark_model_performance(targets, targets)
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0


def test_all_wrong_predictions_score_zero():
    targets = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([1, 1, 1, 0, 0, 0])
    result = benchmark_model_performance(predictions, targets)
    assert result["accuracy"] == 0.0


def test_accuracy_counts_matching_entries():
    targets = np.array([0, 1, 2, 3])
    predictions = np.array([0, 1, 2, 0])
    result = benchmark_model_performance(predictions, targets)
    assert result["accuracy"] == 0.75


def test_confusion_matrix_is_square_over_classes():
    targets = np.array([0, 1, 2, 0, 1, 2])
    result = benchmark_model_performance(targets, targets)
    assert result["confusion_matrix"].shape == (3, 3)
    # Perfect predictions put every count on the diagonal.
    assert np.array_equal(
        result["confusion_matrix"], np.diag(result["confusion_matrix"].diagonal())
    )


def test_balanced_accuracy_differs_on_imbalanced_classes():
    # Nine samples of class 0 predicted correctly, one of class 1 missed.
    targets = np.array([0] * 9 + [1])
    predictions = np.array([0] * 10)
    result = benchmark_model_performance(predictions, targets)
    assert result["accuracy"] == 0.9
    assert result["balanced_accuracy"] == 0.5


def test_label_dict_names_appear_in_report():
    targets = np.array([0, 1, 0, 1])
    result = benchmark_model_performance(targets, targets, label_dict={0: "swim", 1: "turn"})
    assert "swim" in result["classification_report"]
    assert "turn" in result["classification_report"]


def test_without_label_dict_report_uses_class_numbers():
    targets = np.array([0, 1, 0, 1])
    result = benchmark_model_performance(targets, targets)
    assert "swim" not in result["classification_report"]


def test_report_is_logged_not_printed(caplog, capsys):
    targets = np.array([0, 1, 0, 1])
    with caplog.at_level(logging.INFO, logger="poser.core.metrics"):
        benchmark_model_performance(targets, targets)
    assert "Accuracy" in caplog.text
    assert capsys.readouterr().out == ""
