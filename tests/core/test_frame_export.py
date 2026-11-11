"""Tests for poser.core.frame_export."""

from __future__ import annotations

import yaml

from poser.core.frame_export import build_yolo_data_yaml


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
