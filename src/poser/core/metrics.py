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
    report_as_dict: bool = False,
) -> Dict:
    """Score predictions against targets and log a classification report.

    Args:
        predictions: Integer class predictions, shape (N,).
        targets: Integer ground-truth labels, shape (N,).
        label_dict: Maps class index to a readable name. None leaves the
            classes numbered.
        report_as_dict: Return the classification report as sklearn's nested
            dict, for a table, instead of as text.

    Returns:
        Keys accuracy, balanced_accuracy, confusion_matrix and
        classification_report.
    """
    accuracy = accuracy_score(targets, predictions)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    # Name the labels explicitly. A label_dict usually covers every class the
    # project defines, while a given run may only contain some of them, and
    # sklearn rejects target_names that outnumber the classes it observes.
    labels = sorted(label_dict) if label_dict is not None else None
    target_names = [label_dict[i] for i in labels] if labels is not None else None
    matrix = confusion_matrix(targets, predictions, labels=labels)
    report = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=report_as_dict,
    )

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
