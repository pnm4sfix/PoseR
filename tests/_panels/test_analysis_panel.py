"""Tests for poser._panels.analysis_panel."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from poser._panels import analysis_panel
from poser._panels.analysis_panel import AnalysisPanel
from poser.core.session import SessionManager
from poser.core.settings import settings

from .conftest import Dialogs


def _panel(session: SessionManager, export_type: str) -> SimpleNamespace:
    """A stand-in panel on frame 4, exporting individual ind1 as export_type."""
    return SimpleNamespace(
        _viewer=SimpleNamespace(dims=SimpleNamespace(current_step=(4, 0, 0))),
        _session=session,
        _export_type_combo=SimpleNamespace(currentText=lambda: export_type),
        _individual_combo=SimpleNamespace(currentText=lambda: "ind1"),
    )


def test_yolo_frame_export_starts_in_data_frames(tmp_path, monkeypatch):
    # The folder dialog used to open wherever napari was started.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer="")  # Cancel
    monkeypatch.setattr(analysis_panel, "QFileDialog", dialogs)

    AnalysisPanel._on_export_frame(_panel(SessionManager(), "YOLO (pose estimation)"))

    assert dialogs.opened == [tmp_path / "data" / "frames"]


def test_classification_frame_is_added_to_the_chosen_file(tmp_path, monkeypatch):
    # Frames always went to classification_frames.h5, in a folder picked each time.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "swims.h5"))
    monkeypatch.setattr(analysis_panel, "QFileDialog", dialogs)
    monkeypatch.setattr(
        analysis_panel,
        "QMessageBox",
        SimpleNamespace(information=lambda *args: None, warning=lambda *args: None),
    )
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    session.activate(0)
    zeros = np.zeros((3, 10))  # (V, T)
    session.active.coords_data = {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}
    panel = _panel(session, "Classification (behaviour)")
    panel._ask_label = lambda: ("swim", True)

    AnalysisPanel._on_export_frame(panel)

    expected = tmp_path / "data" / "frames" / "classification_frames.h5"
    assert dialogs.opened == [expected]
    assert (tmp_path / "swims.h5").is_file()
