"""Tests for poser.core.batch.

These pin the behaviour that must survive the split of BatchJob into a config
dataclass and a runner. They deliberately do not assert the known Tier-1 bugs
(config is not a dataclass field, mode strings disagree with both callers,
progress_callback arity, missing BatchResult.success, ModelRegistry does not
exist) — those are fixed in later commits and get their own tests.
"""

from __future__ import annotations

import csv
import sys

from poser.core.batch import BatchJob

MANIFEST_COLUMNS = ["pose_file", "video_file", "output", "status", "error"]


def _manifest_rows(output_dir) -> list[dict]:
    with open(output_dir / "batch_manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_empty_job_returns_no_results(tmp_path):
    assert BatchJob(pose_files=[], output_dir=str(tmp_path)).run() == []


def test_empty_job_still_writes_a_manifest(tmp_path):
    BatchJob(pose_files=[], output_dir=str(tmp_path)).run()
    manifest = tmp_path / "batch_manifest.csv"
    assert manifest.exists()
    assert _manifest_rows(tmp_path) == []


def test_manifest_column_order(tmp_path):
    BatchJob(pose_files=[], output_dir=str(tmp_path)).run()
    with open(tmp_path / "batch_manifest.csv", newline="") as f:
        assert next(csv.reader(f)) == MANIFEST_COLUMNS


def test_unreadable_file_is_captured_not_raised(tmp_path):
    results = BatchJob(
        pose_files=[str(tmp_path / "missing.h5")], output_dir=str(tmp_path)
    ).run()
    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].error
    assert results[0].output_path == ""


def test_failure_is_about_the_file_not_a_programming_error(tmp_path):
    # The per-file except catches everything, so a NameError or TypeError in
    # the decode path would be recorded as an ordinary file failure. Assert on
    # the cause, not just that something failed.
    results = BatchJob(
        pose_files=[str(tmp_path / "missing.h5")], output_dir=str(tmp_path)
    ).run()
    error = results[0].error
    assert "not defined" not in error
    assert "positional argument" not in error
    assert "missing.h5" in error or "no such file" in error.lower()


def test_each_input_produces_one_manifest_row(tmp_path):
    pose_files = [str(tmp_path / f"missing{i}.h5") for i in range(3)]
    results = BatchJob(pose_files=pose_files, output_dir=str(tmp_path)).run()
    rows = _manifest_rows(tmp_path)
    assert len(results) == 3
    assert len(rows) == 3
    assert [r["pose_file"] for r in rows] == pose_files


def test_one_failure_does_not_stop_the_rest(tmp_path):
    results = BatchJob(
        pose_files=[str(tmp_path / f"missing{i}.h5") for i in range(3)],
        output_dir=str(tmp_path),
    ).run()
    assert [r.status for r in results] == ["error", "error", "error"]


def test_decode_path_does_not_import_ultralytics(tmp_path):
    # The missing-file guard must stay ahead of the lazy ultralytics import,
    # otherwise this module cannot be tested without a GPU or model weights.
    sys.modules.pop("ultralytics", None)
    BatchJob(
        pose_files=[str(tmp_path / "missing.h5")], output_dir=str(tmp_path)
    ).run()
    assert "ultralytics" not in sys.modules


def test_results_carry_the_input_paths(tmp_path):
    pose = str(tmp_path / "missing.h5")
    results = BatchJob(pose_files=[pose], output_dir=str(tmp_path)).run()
    assert results[0].pose_path == pose
    assert results[0].video_path == ""
