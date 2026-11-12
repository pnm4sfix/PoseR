"""Tests for poser.cli.main."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from poser import api
from poser.cli.main import app
from poser.core.settings import settings
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
