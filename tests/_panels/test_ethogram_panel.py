"""Tests for poser._panels.ethogram_panel."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np

from poser._panels import ethogram_panel
from poser._panels.ethogram_panel import EthogramPanel


def _messages(caplog, level: int) -> list[str]:
    """What the ethogram panel logged at level, in order."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "poser._panels.ethogram_panel" and record.levelno == level
    ]


def test_received_annotations_are_logged_with_frames_per_label(caplog):
    # The Annotation panel sends -1 for the frames of a bout not labelled yet,
    # which the ethogram draws grey, and 0 for the frames outside any bout.
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = SimpleNamespace(_label_names={}, _refresh=lambda: None)

    EthogramPanel.load_annotations(panel, np.array([0, -1, -1, -1, 1, 1, 0]))

    [message] = _messages(caplog, logging.INFO)
    assert "received annotations of shape (7,)" in message
    assert "frames per label {-1: 3, 0: 2, 1: 2}" in message
    assert "0 also covers frames outside any bout" in message


def test_browsed_annotations_and_their_class_names_are_logged(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="poser")
    labels = tmp_path / "fish1_labels.npy"
    np.save(labels, np.array([0, 0, 1]))
    schema = tmp_path / "fish1_schema.json"
    schema.write_text(json.dumps({"0": "other", "1": "swim"}))
    monkeypatch.setattr(
        ethogram_panel,
        "QFileDialog",
        SimpleNamespace(getOpenFileName=lambda *args: (str(labels), "")),
    )
    panel = SimpleNamespace(
        _label_names={},
        _ann_label=SimpleNamespace(setText=lambda text: None),
        _legend_widget=SimpleNamespace(update_names=lambda names: None),
        _refresh=lambda: None,
    )
    panel._try_load_schema = lambda path: EthogramPanel._try_load_schema(panel, path)

    EthogramPanel._on_browse_annotations(panel)

    loaded, names = _messages(caplog, logging.INFO)
    assert f"Loaded annotations from {labels}: shape (3,)" in loaded
    assert "frames per label {0: 2, 1: 1}" in loaded
    assert names == f"Loaded class names from {schema}: {{0: 'other', 1: 'swim'}}"


def test_refresh_without_a_canvas_warns_that_nothing_is_drawn(caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = SimpleNamespace(
        _canvas=None,
        _annotations=np.zeros(5, dtype=np.int64),
        _predictions=None,
        _label_names={},
    )

    EthogramPanel._refresh(panel)

    [message] = _messages(caplog, logging.WARNING)
    assert "No matplotlib canvas" in message
