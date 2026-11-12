"""Tests for poser.core.frame_export."""

from __future__ import annotations

import numpy as np
import yaml

from poser.core.frame_export import (
    build_yolo_data_yaml,
    load_classification_frames,
    save_frame_as_classification,
)


def test_build_yolo_data_yaml_writes_val_dir(tmp_path):
    val_dir = tmp_path / "val" / "images"

    yaml_path = build_yolo_data_yaml(tmp_path, num_keypoints=9, val_dir=val_dir)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    assert data["train"] == "images"
    assert data["val"] == str(val_dir)


def test_build_yolo_data_yaml_reuses_train_images_without_val_dir(tmp_path):
    yaml_path = build_yolo_data_yaml(tmp_path, num_keypoints=9)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    assert data["val"] == "images"


def test_classification_frame_is_appended_to_the_named_file(tmp_path):
    zeros = np.zeros((3, 10))  # (V, T)
    coords_data = {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}

    h5_path = save_frame_as_classification(
        coords_data, "ind1", 4, "swim", "rec_DLC.h5", str(tmp_path),
        file_name="swims.h5",
    )

    assert h5_path == str(tmp_path / "swims.h5")
    poses, _ = load_classification_frames(h5_path)
    assert poses.shape == (1, 3, 3)  # (N, C, V)
