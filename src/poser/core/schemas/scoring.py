"""Data models for scoring a behaviour decoder against hand labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LabelledBout:
    """One hand-labelled bout from a classification .h5."""

    start: int  # first frame
    stop: int  # one past the last frame
    name: str  # class name, e.g. "right"


@dataclass(frozen=True, slots=True)
class ClassNames:
    """A decoder's class names, and the classes its training left out."""

    names: dict[int, str]  # decoder output index -> class name
    ignored: frozenset[str]  # labels_to_ignore: never trained on, so not scored


@dataclass(frozen=True, slots=True)
class DecoderScore:
    """How well per-frame predictions match hand-labelled bouts."""

    accuracy: float
    balanced_accuracy: float
    confusion_matrix: np.ndarray  # (K, K) counts, rows true, columns predicted
    class_names: tuple[str, ...]  # the K classes, in matrix order
    per_class: dict[str, dict[str, float]]  # precision, recall, f1-score, support
    n_scored: int  # bouts that went into the scores
    n_ignored: int  # bouts of a class in labels_to_ignore
    n_skipped: int  # bouts of an unknown class, or with no predicted frames
