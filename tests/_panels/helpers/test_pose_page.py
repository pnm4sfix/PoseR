"""Tests for poser._panels.helpers.pose_page."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from poser._panels.helpers import pose_page
from poser._panels.helpers.pose_page import PosePage, _VideoResult
from poser.core.io import read_poser_coords
from poser.core.session import SessionManager
from poser.core.settings import settings

from ..conftest import Dialogs


class _StubEstimator:
    """Stands in for PoseEstimator, so no video is decoded and no model loads."""

    def model_load(self, path, device):
        pass

    def input_load(self, path, camera=0):
        return 2

    def iter_keypoints(self, **options):
        return iter(())

    def coords_from_keypoints(self, records):
        zeros = np.zeros((3, 2))
        return {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}


def _page(session: SessionManager) -> SimpleNamespace:
    """A stand-in page whose _run records what it was asked to run."""
    page = SimpleNamespace(
        _session=session, _status=SimpleNamespace(append=lambda text: None), runs=[]
    )
    page._run = lambda entries, outputs: page.runs.append((entries, outputs))
    return page


def _estimate(entries, outputs) -> list[_VideoResult]:
    """Run the page's worker on a stub estimator and keep one result per video."""
    page = SimpleNamespace(_estimator=_StubEstimator())
    items = PosePage._estimate_videos(page, entries, outputs, "model.pt", 0, {})
    return [item for item in items if isinstance(item, _VideoResult)]


def test_run_active_asks_for_a_file_starting_in_data_pose(tmp_path, monkeypatch):
    # The pose file used to land beside the video, under a fixed name.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "fish1.h5"))
    monkeypatch.setattr(pose_page, "QFileDialog", dialogs)
    session = SessionManager()
    session.add("", tmp_path / "videos" / "rec.avi")
    session.activate(0)
    page = _page(session)

    PosePage._on_run_active_clicked(page)

    assert dialogs.opened == [tmp_path / "data" / "pose" / "rec.avi_poser_coords.h5"]
    assert page.runs == [([session.active], [tmp_path / "fish1.h5"])]


def test_run_all_saves_each_video_in_the_chosen_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "out"))
    monkeypatch.setattr(pose_page, "QFileDialog", dialogs)
    session = SessionManager()
    session.add("", tmp_path / "a.avi")
    session.add("", tmp_path / "b.avi")
    page = _page(session)

    PosePage._on_run_all_clicked(page)

    assert dialogs.opened == [tmp_path / "data" / "pose"]
    [(_, outputs)] = page.runs
    assert outputs == [
        tmp_path / "out" / "a.avi_poser_coords.h5",
        tmp_path / "out" / "b.avi_poser_coords.h5",
    ]


def test_cancelling_the_dialog_runs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(pose_page, "QFileDialog", Dialogs(answer=""))
    session = SessionManager()
    session.add("", tmp_path / "rec.avi")
    session.activate(0)
    page = _page(session)

    PosePage._on_run_active_clicked(page)

    assert page.runs == []


def test_pose_file_is_written_where_the_user_chose(tmp_path):
    session = SessionManager()
    session.add("", tmp_path / "rec.avi")
    output = tmp_path / "fish1.h5"

    (result,) = _estimate(session.entries, [output])

    assert result.error is None
    assert result.pose_path == str(output)
    assert list(read_poser_coords(output)) == ["ind1"]


def test_second_video_with_the_same_file_name_is_refused(tmp_path):
    # Videos called rec.avi in two folders map to one pose file in the chosen
    # folder, so the second would silently replace the first.
    session = SessionManager()
    session.add("", tmp_path / "day1" / "rec.avi")
    session.add("", tmp_path / "day2" / "rec.avi")
    output = tmp_path / "rec.avi_poser_coords.h5"

    first, second = _estimate(session.entries, [output, output])

    assert first.error is None
    assert "rec.avi_poser_coords.h5" in str(second.error)
