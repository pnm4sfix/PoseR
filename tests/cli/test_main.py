"""Tests for poser.cli.main."""

from __future__ import annotations

from typer.testing import CliRunner

from poser.cli.main import app
from poser.core.settings import settings
from poser.training import data_prep

runner = CliRunner()


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
