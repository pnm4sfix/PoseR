"""Tests for poser._panels.data_panel."""

from __future__ import annotations

import json
from types import SimpleNamespace

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
