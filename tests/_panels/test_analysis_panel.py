"""Tests for poser._panels.analysis_panel."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from poser._panels import analysis_panel
from poser._panels.analysis_panel import AnalysisPanel
from poser.core import bout_detection
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


def _recording(tmp_path, frames: int) -> SessionManager:
    """A session whose active entry holds ind1: 3 nodes over the given frames."""
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    session.activate(0)
    zeros = np.zeros((3, frames))  # (V, T)
    session.active.coords_data = {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}
    return session


def _bout_panel(session: SessionManager) -> SimpleNamespace:
    """A stand-in panel on frame 8, with the default bout detection parameters."""
    panel = SimpleNamespace(
        _viewer=SimpleNamespace(dims=SimpleNamespace(current_step=(8, 0, 0))),
        _session=session,
        _individual_combo=SimpleNamespace(
            currentText=lambda: "ind1", count=lambda: 1, itemText=lambda i: "ind1"
        ),
        _method_combo=SimpleNamespace(currentText=lambda: "orthogonal"),
        _fps_spin=SimpleNamespace(value=lambda: 25),
        _center_node_spin=SimpleNamespace(value=lambda: 0),
        _thresh_spin=SimpleNamespace(value=lambda: 5.0),
        _min_frames_spin=SimpleNamespace(value=lambda: 10),
        _max_frames_spin=SimpleNamespace(value=lambda: 300),
        _manual_start=None,
        _detected_bouts=[],
        _on_bouts_updated=None,
        _update_bout_count=lambda: None,
        _manual_range_label=SimpleNamespace(setText=lambda text: None),
        _btn_set_end=SimpleNamespace(setEnabled=lambda enabled: None),
    )
    panel._save_bouts_to_session = lambda bouts: (
        AnalysisPanel._save_bouts_to_session(panel, bouts)
    )
    return panel


def _messages(caplog, level: int) -> list[str]:
    """What the analysis panel logged at level."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "poser._panels.analysis_panel" and record.levelno == level
    ]


def test_detection_logs_its_parameters_and_the_labelled_bouts_it_replaces(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="poser")
    # The second bout is under 10 frames long and the third over 300.
    found = [(2, 20), (30, 35), (40, 400)]
    monkeypatch.setattr(
        bout_detection,
        "orthogonal_variance",
        lambda points, center_node, fps, n_nodes: (found, None, None, None),
    )
    session = _recording(tmp_path, frames=500)
    # Labelled in the Annotation panel before detection was run again.
    session.active.bouts = [{"start": 2, "end": 20, "label": 1}]

    AnalysisPanel._on_detect(_bout_panel(session))

    [message] = _messages(caplog, logging.INFO)
    assert f"Detected 1 bouts on {tmp_path / 'rec_DLC.h5'} " in message
    assert "method orthogonal, fps 25, centre node 0, 10-300 frames long" in message
    assert (
        f"Stored 1 bouts (0 labelled) on session entry {tmp_path / 'rec_DLC.h5'}; "
        "it had 1 labelled before"
    ) in _messages(caplog, logging.DEBUG)


def test_added_manual_bout_is_logged(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = _bout_panel(_recording(tmp_path, frames=10))
    panel._manual_start = 2

    AnalysisPanel._on_set_end(panel)  # the viewer is on frame 8

    [message] = _messages(caplog, logging.INFO)
    assert "Added manual bout 2-8 for ind1 on " in message
    assert "1 bouts in total" in message


def test_detection_without_pose_data_logs_why_nothing_happened(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    shown = []
    monkeypatch.setattr(
        analysis_panel,
        "QMessageBox",
        SimpleNamespace(warning=lambda *args: shown.append(args[-1])),
    )

    AnalysisPanel._on_detect(_bout_panel(SessionManager()))

    [message] = _messages(caplog, logging.WARNING)
    assert "Bout detection skipped: no pose data loaded" in message
    assert shown == ["Load a pose file first."]
