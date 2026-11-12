"""Tests for poser._panels.helpers.decode_page."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from poser._panels.helpers import decode_page
from poser._panels.helpers.decode_page import DecodePage, _FileResult
from poser.core.session import SessionManager
from poser.core.settings import settings

from ..conftest import Dialogs


class _StubInference:
    """Stands in for Inference: two frames of class 1, recording where each saves."""

    n_frames = 2

    def __init__(self) -> None:
        self.saved: list[Path] = []

    def input_load(self, path, coords_data):
        return "stub pose"

    def input_adapt(self):
        return []

    def iter_predictions(self):
        yield np.array([1, 1])

    def predictions_save(self, predictions, path):
        self.saved.append(path)
        return path


def _page(session: SessionManager) -> SimpleNamespace:
    """A stand-in page with a decoder loaded, whose _run records what it was asked."""
    page = SimpleNamespace(
        _session=session,
        _checkpoint=Path("decoder.ckpt"),
        _status=SimpleNamespace(append=lambda text: None),
        runs=[],
    )
    page._run = lambda entries, outputs: page.runs.append((entries, outputs))
    return page


def _predict(entries, outputs) -> tuple[SimpleNamespace, list[_FileResult]]:
    """Run the page's worker on a stub decoder and keep one result per file."""
    page = SimpleNamespace(_inference=_StubInference())
    items = DecodePage._predict(page, entries, outputs)
    return page, [item for item in items if isinstance(item, _FileResult)]


def test_predict_asks_for_a_file_starting_in_data_predictions(tmp_path, monkeypatch):
    # Predictions used to land beside the pose file, under a fixed name.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "fish1.npy"))
    monkeypatch.setattr(decode_page, "QFileDialog", dialogs)
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    session.activate(0)
    page = _page(session)

    DecodePage._on_predict_clicked(page)

    expected = tmp_path / "data" / "predictions" / "rec_DLC_predictions.npy"
    assert dialogs.opened == [expected]
    assert page.runs == [([session.active], [tmp_path / "fish1.npy"])]


def test_predict_all_saves_each_file_in_the_chosen_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "out"))
    monkeypatch.setattr(decode_page, "QFileDialog", dialogs)
    session = SessionManager()
    session.add(tmp_path / "a_DLC.h5")
    session.add(tmp_path / "b_DLC.h5")
    page = _page(session)

    DecodePage._on_predict_all_clicked(page)

    assert dialogs.opened == [tmp_path / "data" / "predictions"]
    [(_, outputs)] = page.runs
    assert outputs == [
        tmp_path / "out" / "a_DLC_predictions.npy",
        tmp_path / "out" / "b_DLC_predictions.npy",
    ]


def test_no_dialog_opens_before_a_decoder_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "fish1.npy"))
    monkeypatch.setattr(decode_page, "QFileDialog", dialogs)
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    session.activate(0)
    page = _page(session)
    page._checkpoint = None

    DecodePage._on_predict_clicked(page)

    assert dialogs.opened == []
    assert page.runs == []


def test_predictions_are_saved_where_the_user_chose(tmp_path):
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    output = tmp_path / "fish1.npy"

    page, (result,) = _predict(session.entries, [output])

    assert result.error is None
    assert result.saved_path == output
    assert page._inference.saved == [output]


def test_second_file_with_the_same_predictions_name_is_refused(tmp_path):
    # Pose files called rec_DLC.h5 in two folders map to one predictions file
    # in the chosen folder, so the second would silently replace the first.
    session = SessionManager()
    session.add(tmp_path / "day1" / "rec_DLC.h5")
    session.add(tmp_path / "day2" / "rec_DLC.h5")
    output = tmp_path / "rec_DLC_predictions.npy"

    page, (first, second) = _predict(session.entries, [output, output])

    assert first.error is None
    assert "rec_DLC_predictions.npy" in str(second.error)
    assert page._inference.saved == [output]
