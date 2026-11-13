"""Tests for poser.cli.main."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from typer.testing import CliRunner

from poser import api
from poser.cli.main import app
from poser.core.batch import BatchRunner
from poser.core.settings import settings
from poser.models.st_gcn_aaai18_pylightning_3block import ST_GCN_18
from poser.training import data_prep

runner = CliRunner()


class _StubPoseR:
    """Stands in for poser.api.PoseR, which predict builds."""

    def __init__(self, config=None):
        pass

    def predict(self, pose_file, checkpoint, individual):
        return {0: 1, 1: 0}


def test_train_writes_runs_under_data_runs_by_default(tmp_path, monkeypatch):
    # Runs used to go to poser_runs/ beside the first training file.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.chdir(tmp_path)  # no decoder_config.yml to pick up
    seen = {}

    def stop_before_training(files, cfg):
        seen["output_dir"] = cfg.output_dir
        raise RuntimeError("stop before training")

    monkeypatch.setattr(data_prep, "prepare_dataset", stop_before_training)

    runner.invoke(app, ["train", str(tmp_path / "rec_pose.npy")])

    assert seen["output_dir"] == tmp_path / "data" / "runs"


def test_predict_writes_under_data_predictions_by_default(tmp_path, monkeypatch):
    # The .json used to land beside the pose file.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api, "PoseR", _StubPoseR)

    result = runner.invoke(
        app, ["predict", str(tmp_path / "rec.h5"), "--checkpoint", "decoder.ckpt"]
    )

    assert result.exit_code == 0, result.output
    written = tmp_path / "data" / "predictions" / "rec.predictions.json"
    assert json.loads(written.read_text()) == {"0": 1, "1": 0}


class _StubDecoder:
    """Stands in for a loaded ST_GCN_18 and labels every window class 1."""

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, windows):
        return torch.tensor([[0.0, 1.0]]).repeat(len(windows), 1)


def test_predict_npy_writes_under_data_predictions_by_default(tmp_path, monkeypatch):
    # The .npy and .csv used to land beside the pose file.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    pose_file = tmp_path / "rec_pose.npy"
    np.save(pose_file, np.random.default_rng(0).normal(size=(3, 12, 9, 1)))
    checkpoint = tmp_path / "decoder.ckpt"
    checkpoint.touch()
    hparams = {"hyper_parameters": {"data_cfg": {"T2": 4}, "num_class": 2}}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: hparams)
    monkeypatch.setattr(
        ST_GCN_18,
        "load_from_checkpoint",
        classmethod(lambda cls, *args, **kwargs: _StubDecoder()),
    )

    result = runner.invoke(
        app,
        ["predict-npy", str(pose_file), "--checkpoint", str(checkpoint), "-d", "cpu"],
    )

    assert result.exit_code == 0, result.output
    saved = tmp_path / "data" / "predictions" / "rec_pose_predictions.npy"
    assert np.load(saved).tolist() == [1] * 12
    assert saved.with_suffix(".csv").is_file()


def test_batch_writes_under_data_batch_by_default(tmp_path, monkeypatch):
    # It used to write to batch_output/ in whatever folder it was started from.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    seen = {}

    def record(batch_runner):
        seen["output_dir"] = batch_runner._job.output_dir
        return []

    monkeypatch.setattr(BatchRunner, "run", record)

    result = runner.invoke(app, ["batch", str(tmp_path / "rec_DLC.h5")])

    assert result.exit_code == 0, result.output
    assert seen["output_dir"] == str(tmp_path / "data" / "batch")


class _FakeYOLO:
    """Stands in for ultralytics.YOLO and records what train() was given."""

    train_kwargs: dict = {}

    def __init__(self, weights):
        pass

    def train(self, **kwargs):
        _FakeYOLO.train_kwargs = kwargs


def test_finetune_writes_under_data_finetune_by_default(tmp_path, monkeypatch):
    # The command passed its own poser_yolo_finetune/ default, relative to the
    # folder it was started from.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FakeYOLO))
    frames = tmp_path / "frames"
    frames.mkdir()

    result = runner.invoke(app, ["finetune", str(frames)])

    assert result.exit_code == 0, result.output
    assert _FakeYOLO.train_kwargs["project"] == str(tmp_path / "data" / "finetune")


def test_model_repair_writes_under_data_models_by_default(tmp_path, monkeypatch):
    # The repaired copy used to land beside the checkpoint.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    checkpoint = tmp_path / "decoder.ckpt"
    decoder = ST_GCN_18(in_channels=3, num_class=4, graph_cfg={"layout": "zebrafish"})
    torch.save({"state_dict": decoder.state_dict()}, checkpoint)
    config = tmp_path / "decoder_config.yml"
    config.write_text(yaml.safe_dump({"data_cfg": {"V": 9, "center": 4, "T2": 100}}))

    result = runner.invoke(
        app, ["model", "repair", str(checkpoint), "--config", str(config)]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "data" / "models" / "decoder.repaired.ckpt").is_file()
