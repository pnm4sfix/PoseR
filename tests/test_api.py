"""Tests for poser.api."""

from __future__ import annotations

from poser.api import PoseR
from poser.core.batch import BatchRunner
from poser.core.settings import settings


def test_batch_writes_under_data_batch_by_default(tmp_path, monkeypatch):
    # It used to write to batch_output/ in whatever folder Python started in.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    seen = {}

    def record(batch_runner):
        seen["output_dir"] = batch_runner._job.output_dir
        return []

    monkeypatch.setattr(BatchRunner, "run", record)

    PoseR().batch([tmp_path / "rec_DLC.h5"])

    assert seen["output_dir"] == str(tmp_path / "data" / "batch")
