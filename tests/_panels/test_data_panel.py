"""Tests for poser._panels.data_panel."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from poser._panels import data_panel
from poser._panels.data_panel import DataPanel
from poser.core.session import SessionManager
from poser.core.settings import settings

from .conftest import Dialogs


def test_save_session_starts_in_data_sessions(tmp_path, monkeypatch):
    # It used to start wherever napari was started.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "day1.json"))
    monkeypatch.setattr(data_panel, "QFileDialog", dialogs)
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")

    DataPanel._on_save_session(SimpleNamespace(_session=session))

    assert dialogs.opened == [tmp_path / "data" / "sessions" / "session.json"]
    saved = json.loads((tmp_path / "day1.json").read_text())
    assert saved["entries"][0]["pose_path"] == str(tmp_path / "rec_DLC.h5")


def test_loading_an_entry_logs_the_pose_file_and_its_points_layer(tmp_path, caplog):
    # The Analysis panel detects bouts on the coords parsed here.
    caplog.set_level(logging.DEBUG, logger="poser")
    columns = pd.MultiIndex.from_product(
        [["scorer1"], ["head", "tail"], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    pose = tmp_path / "rec_DLC.csv"
    pd.DataFrame(np.ones((4, 6)), columns=columns).to_csv(pose)  # 4 frames, 2 nodes
    entry = SessionManager().add(pose)
    added = []
    viewer = SimpleNamespace(
        layers=[], add_points=lambda points, **kwargs: added.append(kwargs["name"])
    )

    DataPanel._load_entry_layers(SimpleNamespace(_viewer=viewer), entry)

    assert added == ["rec_DLC_points"]
    loaded, layer = [
        record.getMessage()
        for record in caplog.records
        if record.name == "poser._panels.data_panel" and record.levelno == logging.INFO
    ]
    assert f"Loaded pose file {pose}: 1 individual(s)" in loaded
    assert "Added points layer rec_DLC_points: 8 points" in layer
    assert (
        "poser._panels.data_panel",
        logging.DEBUG,
        "DLC levels ['scorer', 'bodyparts', 'coords']: 'individuals' lookup failed; "
        "reading it as one animal",
    ) in caplog.record_tuples


def test_an_unreadable_pose_file_is_logged_with_its_traceback(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="poser")
    shown = []
    monkeypatch.setattr(
        data_panel,
        "QMessageBox",
        SimpleNamespace(warning=lambda *args: shown.append(args[-1])),
    )
    pose = tmp_path / "rec_DLC.csv"
    pose.write_text("not a pose table\n")
    entry = SessionManager().add(pose)

    DataPanel._load_entry_layers(
        SimpleNamespace(_viewer=SimpleNamespace(layers=[])), entry
    )

    assert shown == [f"Could not parse pose file: {pose}"]
    [error] = [
        record
        for record in caplog.records
        if record.name == "poser._panels.data_panel" and record.levelno == logging.ERROR
    ]
    assert f"Could not load pose file {pose}" in error.getMessage()
    assert error.exc_info is not None


def test_a_folder_without_pose_files_logs_why_nothing_was_added(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="poser")
    monkeypatch.setattr(data_panel, "QFileDialog", Dialogs(answer=str(tmp_path)))
    shown = []
    monkeypatch.setattr(
        data_panel,
        "QMessageBox",
        SimpleNamespace(information=lambda *args: shown.append(args[-1])),
    )
    session = SessionManager()

    DataPanel._on_add_folder(SimpleNamespace(_session=session))

    assert shown == ["No .h5/.hdf5/.csv files found."]
    assert session.entries == []
    [warning] = [
        record
        for record in caplog.records
        if record.name == "poser._panels.data_panel"
        and record.levelno == logging.WARNING
    ]
    assert f"under {tmp_path}; nothing added" in warning.getMessage()


def test_activating_an_entry_sends_it_to_the_other_panels(tmp_path):
    # The Annotation panel kept listing and labelling the previous file's bouts.
    session = SessionManager()
    session.add(tmp_path / "fish1_DLC.h5")
    session.add(tmp_path / "fish2_DLC.h5")
    zeros = np.zeros((3, 30))  # (V, T)
    sent = []
    panel = SimpleNamespace(
        _session=session,
        _session_list=SimpleNamespace(currentRow=lambda: 1),
        _load_entry_layers=lambda entry: setattr(
            entry, "coords_data", {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}
        ),
        _refresh_table=lambda: None,
        active_entry_changed=SimpleNamespace(
            emit=lambda entry: sent.append((entry, bool(entry.coords_data)))
        ),
    )

    DataPanel._on_activate(panel)
    DataPanel._on_next_pending(panel)  # fish1 is still pending

    # Sent after the layers load, as the Annotation panel takes T from the coords.
    assert sent == [(session.entries[1], True), (session.entries[0], False)]


def test_removing_an_entry_sends_the_active_entry_or_none(tmp_path):
    session = SessionManager()
    session.add(tmp_path / "fish1_DLC.h5")
    fish2 = session.add(tmp_path / "fish2_DLC.h5")
    session.activate(1)
    sent = []
    panel = SimpleNamespace(
        _session=session,
        _session_list=SimpleNamespace(currentRow=lambda: 0),
        _refresh_table=lambda: None,
        active_entry_changed=SimpleNamespace(emit=sent.append),
    )

    DataPanel._on_remove(panel)  # fish1; the active fish2 moves up to row 0
    DataPanel._on_remove(panel)  # fish2, so no entry is active

    assert sent == [fish2, None]


def test_loading_a_session_sends_its_active_entry(tmp_path, monkeypatch):
    saved = SessionManager()
    saved.add(tmp_path / "fish1_DLC.h5")
    saved.add(tmp_path / "fish2_DLC.h5")
    saved.activate(1)
    saved.save(tmp_path / "day1.json")
    chosen = (str(tmp_path / "day1.json"), "")
    monkeypatch.setattr(
        data_panel, "QFileDialog", SimpleNamespace(getOpenFileName=lambda *a: chosen)
    )
    session = SessionManager()
    sent = []
    panel = SimpleNamespace(
        _session=session,
        _refresh_table=lambda: None,
        active_entry_changed=SimpleNamespace(emit=sent.append),
    )

    DataPanel._on_load_session(panel)

    assert sent == [session.entries[1]]
    assert sent[0].pose_path == str(tmp_path / "fish2_DLC.h5")
