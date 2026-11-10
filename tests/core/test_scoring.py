"""Tests for poser.core.scoring."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from poser.core.schemas.scoring import ClassNames, LabelledBout
from poser.core.scoring import (
    class_names_from_checkpoint,
    class_names_from_config,
    labelled_bouts,
    save_score_csv,
    score_bouts,
)

CLASSES = ClassNames(
    names={0: "forward", 1: "left", 2: "right", 3: "unclassified"},
    ignored=frozenset({"unclassified"}),
)


def test_labelled_bouts_covers_every_individual():
    data = {
        1: {1: {"start": 0, "stop": 5, "classification": "right"}},
        2: {1: {"start": 7, "stop": 9, "classification": "left"}},
    }
    assert labelled_bouts(data) == [
        LabelledBout(0, 5, "right"),
        LabelledBout(7, 9, "left"),
    ]


def test_bout_stop_is_exclusive():
    # Frames 0-2 vote right. Counting frame 3 as well would tie and pick forward.
    predictions = np.array([0, 2, 2, 0, 0, 0])
    score = score_bouts(predictions, [LabelledBout(0, 3, "right")], CLASSES)
    assert score.accuracy == 1.0


def test_ignored_classes_are_left_out():
    predictions = np.full(10, 2)
    bouts = [LabelledBout(0, 5, "right"), LabelledBout(5, 10, "unclassified")]
    score = score_bouts(predictions, bouts, CLASSES)
    assert (score.n_scored, score.n_ignored) == (1, 1)
    assert "unclassified" not in score.class_names
    assert score.confusion_matrix.shape == (3, 3)


def test_known_confusion_matrix():
    predictions = np.array([0] * 10 + [2] * 10 + [0] * 10)
    bouts = [
        LabelledBout(0, 10, "forward"),
        LabelledBout(10, 20, "right"),
        LabelledBout(20, 30, "right"),
    ]
    score = score_bouts(predictions, bouts, CLASSES)
    assert score.class_names == ("forward", "left", "right")
    np.testing.assert_array_equal(
        score.confusion_matrix, [[1, 0, 0], [0, 0, 0], [1, 0, 1]]
    )
    assert score.accuracy == pytest.approx(2 / 3)
    assert score.balanced_accuracy == pytest.approx((1.0 + 0.5) / 2)
    assert score.per_class["right"]["recall"] == pytest.approx(0.5)


def test_unknown_and_empty_bouts_are_skipped():
    predictions = np.full(10, 2)
    bouts = [
        LabelledBout(0, 5, "right"),
        LabelledBout(0, 5, "jump"),
        LabelledBout(50, 60, "right"),
    ]
    score = score_bouts(predictions, bouts, CLASSES)
    assert (score.n_scored, score.n_skipped) == (1, 2)


def test_no_scoreable_bout_raises():
    with pytest.raises(ValueError):
        score_bouts(np.zeros(2), [LabelledBout(0, 2, "unclassified")], CLASSES)


def test_config_accepts_number_and_name_forms(tmp_path):
    path = tmp_path / "decoder_config.yml"
    path.write_text(
        "data_cfg:\n"
        "  classification_dict:\n"
        "    0: forward\n"
        "    '1': left\n"
        "    2: right\n"
        "  labels_to_ignore: [forward, 1]\n"
    )
    classes = class_names_from_config(path)
    assert classes.names == {0: "forward", 1: "left", 2: "right"}
    assert classes.ignored == frozenset({"forward", "left"})


def test_checkpoint_names_only_when_stored_as_names():
    repaired = {
        "hyper_parameters": {
            "data_cfg": {
                "label_dict": {0: "forward", 1: "right"},
                "labels_to_ignore": ["right"],
            }
        }
    }
    classes = class_names_from_checkpoint(repaired)
    assert classes.names == {0: "forward", 1: "right"}
    assert classes.ignored == frozenset({"right"})

    numbers_only = {"hyper_parameters": {"data_cfg": {"label_dict": {0: 0, 1: 1}}}}
    assert class_names_from_checkpoint(numbers_only) is None
    assert class_names_from_checkpoint({}) is None


def test_save_score_csv_writes_metrics_and_matrix(tmp_path):
    predictions = np.array([0] * 10 + [2] * 10)
    bouts = [LabelledBout(0, 10, "forward"), LabelledBout(10, 20, "right")]
    score = score_bouts(predictions, bouts, CLASSES)
    metrics_path, matrix_path = save_score_csv(
        score, tmp_path / "rec_confusion_matrix.png"
    )

    assert metrics_path.name == "rec_confusion_matrix_metrics.csv"
    with open(metrics_path) as fh:
        metrics = list(csv.reader(fh))
    assert metrics[0] == ["class", "precision", "recall", "f1", "support"]
    assert [row[0] for row in metrics[1:]] == [
        "forward", "left", "right", "accuracy", "balanced accuracy",
    ]

    with open(matrix_path) as fh:
        matrix = list(csv.reader(fh))
    assert matrix[0] == ["true \\ predicted", "forward", "left", "right"]
    assert matrix[3] == ["right", "0", "0", "1"]
