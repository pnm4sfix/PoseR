"""Classification metrics for behaviour decoding."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

log = logging.getLogger(__name__)


def benchmark_model_performance(
    predictions: np.ndarray,
    targets: np.ndarray,
    label_dict: Optional[Dict[int, str]] = None,
) -> Dict:
    """Score predictions against targets and log a classification report.

    Args:
        predictions: Integer class predictions, shape (N,).
        targets: Integer ground-truth labels, shape (N,).
        label_dict: Maps class index to a readable name. None leaves the
            classes numbered.

    Returns:
        Keys accuracy, balanced_accuracy, confusion_matrix and
        classification_report.
    """
    accuracy = accuracy_score(targets, predictions)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    matrix = confusion_matrix(targets, predictions)
    target_names = (
        [label_dict[i] for i in sorted(label_dict.keys())]
        if label_dict is not None
        else None
    )
    report = classification_report(targets, predictions, target_names=target_names)

    log.info(
        "Accuracy: %.4f | Balanced accuracy: %.4f", accuracy, balanced_accuracy
    )
    log.info("Classification report:\n%s", report)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": matrix,
        "classification_report": report,
    }
