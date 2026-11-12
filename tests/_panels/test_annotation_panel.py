"""Tests for poser._panels.annotation_panel."""

from __future__ import annotations

import json
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
