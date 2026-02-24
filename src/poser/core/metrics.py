"""
Model evaluation metrics — napari-free.

Extracted from ``PoserWidget.benchmark_model_performance()``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def benchmark_model_performance(
    model,
    predictions: np.ndarray,
    targets: np.ndarray,
    label_dict: Optional[Dict[int, str]] = None,
) -> Dict:
    """Compute classification metrics and optionally print a report.

    Parameters
    ----------
    model:
        Not used directly but kept for API compatibility with widget call-site.
    predictions:
        Shape ``(N,)`` — integer class predictions.
    targets:
        Shape ``(N,)`` — integer class ground-truth labels.
    label_dict:
        Optional ``{int: str}`` mapping for human-readable class names.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``balanced_accuracy``, ``confusion_matrix``,
        ``classification_report``.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        classification_report,
    )

    acc = accuracy_score(targets, predictions)
    bal_acc = balanced_accuracy_score(targets, predictions)
    cm = confusion_matrix(targets, predictions)
    target_names = (
        [label_dict[i] for i in sorted(label_dict.keys())]
        if label_dict is not None
        else None
    )
    report = classification_report(targets, predictions, target_names=target_names)

    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced accuracy: {bal_acc:.4f}")
    print(report)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
