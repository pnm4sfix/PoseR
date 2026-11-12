"""Tests for poser.core.inference."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from poser.core.inference import Inference, predictions_file_name


@pytest.mark.parametrize(
    ("pose_path", "name"),
    [
        ("videos/rec_DLC.h5", "rec_DLC_predictions.npy"),
        ("training/rec_pose.npy", "rec_predictions.npy"),
        (
            "pose/test_video.avi_poser_coords.h5",
            "test_video.avi_poser_coords_predictions.npy",
        ),
    ],
)
def test_predictions_are_named_after_the_pose_file(pose_path, name):
    assert predictions_file_name(pose_path) == name


def test_predictions_save_writes_where_it_is_told(tmp_path):
    # They used to go beside the pose file, under a name nobody could change.
    predictions = np.array([0, 2, 1])

    saved = Inference().predictions_save(predictions, tmp_path / "fish1.npy")

    assert saved == tmp_path / "fish1.npy"
    np.testing.assert_array_equal(np.load(saved), predictions)
    with open(tmp_path / "fish1.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows == [["frame", "predicted_label"], ["0", "0"], ["1", "2"], ["2", "1"]]


def test_predictions_save_keeps_a_name_without_npy(tmp_path):
    # np.save given a path would add .npy, leaving the returned path wrong.
    saved = Inference().predictions_save(np.array([1]), tmp_path / "fish1")

    assert saved.is_file()
