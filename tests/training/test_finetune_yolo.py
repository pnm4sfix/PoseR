"""Tests for poser.training.finetune_yolo."""

from __future__ import annotations

import sys
import types

import yaml

from poser.training.finetune_yolo import finetune_yolo


class _FakeYOLO:
    """Stands in for ultralytics.YOLO and records what train() was given."""

    train_kwargs: dict = {}

    def __init__(self, weights):
        pass

    def train(self, **kwargs):
        _FakeYOLO.train_kwargs = kwargs


def test_finetune_yolo_generates_data_yaml_with_val_dir(tmp_path, monkeypatch):
    # finetune_yolo passed val_dir to a builder that did not take it, so every
    # call without a ready-made data.yaml raised TypeError.
    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO)
    )
    val_dir = tmp_path / "val" / "images"

    finetune_yolo(tmp_path, val_dir=val_dir, project=str(tmp_path / "runs"))

    with open(_FakeYOLO.train_kwargs["data"]) as f:
        data = yaml.safe_load(f)
    assert data["val"] == str(val_dir)


def test_extra_kwargs_override_named_defaults(tmp_path, monkeypatch):
    # A key such as batch reached dict() twice, once named and once through
    # **extra_kwargs, which raised TypeError.
    monkeypatch.setitem(
        sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO)
    )

    finetune_yolo(
        tmp_path,
        data_yaml=tmp_path / "data.yaml",
        project=str(tmp_path / "project"),
        extra_kwargs={"batch": 8, "patience": 5},
    )

    assert _FakeYOLO.train_kwargs["batch"] == 8
    assert _FakeYOLO.train_kwargs["patience"] == 5
    assert _FakeYOLO.train_kwargs["epochs"] == 50
