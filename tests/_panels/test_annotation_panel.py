"""Tests for poser._panels.annotation_panel."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np

from poser._panels import annotation_panel
from poser._panels.annotation_panel import AnnotationPanel
from poser.core.session import SessionManager
from poser.core.settings import settings

from .conftest import Dialogs


def _panel(tmp_path) -> SimpleNamespace:
    """A stand-in panel: one bout labelled swim on a 10-frame, 3-node recording."""
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    session.activate(0)
    zeros = np.zeros((3, 10))  # (V, T)
    session.active.coords_data = {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}
    return SimpleNamespace(
        _session=session,
        _current_bouts=[{"start": 2, "end": 5, "label": 1}],
        _behaviour_labels={0: "other", 1: "swim"},
    )


def test_training_data_is_saved_under_the_chosen_name(tmp_path, monkeypatch):
    # The files were named after the pose file, in a folder picked each time.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "fish1_pose.npy"))
    monkeypatch.setattr(annotation_panel, "QFileDialog", dialogs)
    shown = []
    monkeypatch.setattr(
        annotation_panel,
        "QMessageBox",
        SimpleNamespace(
            information=lambda *args: shown.append(args[-1]),
            warning=lambda *args: shown.append(args[-1]),
            critical=lambda *args: shown.append(args[-1]),
        ),
    )

    AnnotationPanel._on_export(_panel(tmp_path))

    assert dialogs.opened == [tmp_path / "data" / "training" / "rec_DLC_pose.npy"]
    assert np.load(tmp_path / "fish1_pose.npy").shape == (3, 10, 3, 1)  # (C, T, V, M)
    labels = np.load(tmp_path / "fish1_labels.npy")
    assert labels.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    schema = json.loads((tmp_path / "fish1_schema.json").read_text())
    assert schema == {"0": "other", "1": "swim"}


def test_name_without_the_pose_suffix_still_pairs_for_training(tmp_path, monkeypatch):
    # The Train panel pairs <stem>_pose.npy with <stem>_labels.npy.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(
        annotation_panel, "QFileDialog", Dialogs(answer=str(tmp_path / "fish1"))
    )
    monkeypatch.setattr(
        annotation_panel, "QMessageBox", SimpleNamespace(information=lambda *a: None)
    )

    AnnotationPanel._on_export(_panel(tmp_path))

    assert (tmp_path / "fish1_pose.npy").is_file()
    assert (tmp_path / "fish1_labels.npy").is_file()


def _labelling_panel(tmp_path, row: int, bouts: list[dict]) -> SimpleNamespace:
    """The stand-in panel listing bouts, with row selected and label 1 clicked."""
    panel = _panel(tmp_path)
    panel._current_bouts = bouts
    panel._bout_list = SimpleNamespace(
        currentRow=lambda: row,
        count=lambda: len(bouts),
        item=lambda r: SimpleNamespace(setText=lambda text: None),
        setCurrentRow=lambda r: None,
    )
    panel.sender = lambda: SimpleNamespace(property=lambda name: 1)
    panel._current_label_display = SimpleNamespace(setText=lambda text: None)
    panel._update_stats = lambda: None
    panel._emit_annotations = lambda: None
    return panel


def _messages(caplog, level: int) -> list[str]:
    """What the annotation panel logged at level."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "poser._panels.annotation_panel" and record.levelno == level
    ]


def test_assigned_label_is_logged_with_its_bout_and_frames(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = _labelling_panel(tmp_path, row=0, bouts=[{"start": 2, "end": 5}])

    AnnotationPanel._on_assign_label(panel)

    [message] = _messages(caplog, logging.INFO)
    assert "Assigned label 1 (swim) to bout 1, frames 2-5" in message


def test_label_click_with_no_bout_selected_logs_a_warning(tmp_path, caplog):
    # Nothing is assigned, so the bouts stay unlabelled.
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = _labelling_panel(tmp_path, row=-1, bouts=[{"start": 2, "end": 5}])

    AnnotationPanel._on_assign_label(panel)

    [message] = _messages(caplog, logging.WARNING)
    assert "Label not assigned: no bout selected (row -1, 1 bouts loaded)" in message
    assert panel._current_bouts == [{"start": 2, "end": 5}]


def test_loaded_bouts_are_logged_with_their_label_counts(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="poser")
    monkeypatch.setattr(annotation_panel, "QListWidgetItem", str)
    panel = _panel(tmp_path)  # a 10-frame recording
    panel._bout_list = SimpleNamespace(clear=lambda: None, addItem=lambda item: None)
    panel._update_stats = lambda: None
    panel._bouts_to_labels = lambda: AnnotationPanel._bouts_to_labels(panel)
    panel._emit_annotations = lambda: AnnotationPanel._emit_annotations(panel)
    panel.annotations_changed = SimpleNamespace(emit=lambda labels: None)
    # Detectors pad bouts, so the last one can run past the end of the recording.
    bouts = [{"start": 2, "end": 5, "label": 1}, {"start": 6, "end": 14}]

    AnnotationPanel.load_bouts(panel, bouts)

    [loaded] = _messages(caplog, logging.INFO)
    assert "Loaded 2 bouts into the annotation list: 1 labelled, 1 unlabelled" in (
        loaded
    )
    [short] = _messages(caplog, logging.WARNING)
    assert "Label array has 10 frames, shorter than the last bout end 14" in short
