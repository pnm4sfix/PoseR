"""Tests for poser._panels.metrics_panel."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from poser._panels import metrics_panel
from poser._panels.metrics_panel import MetricsPanel
from poser.core.schemas.scoring import ClassNames, DecoderScore, LabelledBout
from poser.core.settings import settings

from .conftest import Dialogs


def _messages(caplog, level: int) -> list[str]:
    """What the metrics panel logged at level, in order."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "poser._panels.metrics_panel" and record.levelno == level
    ]


def test_export_starts_in_data_metrics_named_after_the_predictions(
    tmp_path, monkeypatch
):
    # It used to start beside the predictions file.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer="")  # Cancel
    monkeypatch.setattr(metrics_panel, "QFileDialog", dialogs)
    panel = SimpleNamespace(_predictions_path=tmp_path / "rec_predictions.npy")

    MetricsPanel._on_export_clicked(panel)

    expected = tmp_path / "data" / "metrics" / "rec_predictions_confusion_matrix.png"
    assert dialogs.opened == [expected]


def test_export_of_predictions_from_inference_starts_in_data_metrics(
    tmp_path, monkeypatch
):
    # Predictions sent by the Inference panel have no file to name the export
    # after; it used to start wherever napari was started.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer="")  # Cancel
    monkeypatch.setattr(metrics_panel, "QFileDialog", dialogs)

    MetricsPanel._on_export_clicked(SimpleNamespace(_predictions_path=None))

    assert dialogs.opened == [tmp_path / "data" / "metrics" / "confusion_matrix.png"]


def test_predictions_from_inference_are_logged_with_frames_per_class(caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = SimpleNamespace(
        _classes_browsed=False,
        _predictions_label=SimpleNamespace(setText=lambda text: None),
        _score_if_ready=lambda: None,
    )

    MetricsPanel.load_predictions(panel, np.array([2, 0, 2, 2]))

    [message] = _messages(caplog, logging.INFO)
    assert message.startswith("Metrics received 4 frames of predictions")
    assert "frames per class {0: 1, 2: 3}" in message


def test_a_score_is_logged_and_bouts_of_unknown_classes_are_flagged(
    monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="poser")
    score = DecoderScore(
        accuracy=1.0,
        balanced_accuracy=1.0,
        confusion_matrix=np.eye(2, dtype=int),
        class_names=("rest", "swim"),
        per_class={},
        n_scored=2,
        n_ignored=0,
        n_skipped=1,
    )
    monkeypatch.setattr(metrics_panel, "score_bouts", lambda *args: score)
    panel = SimpleNamespace(
        _predictions=np.array([0, 0, 0, 1, 1, 1, 0, 0]),
        _bouts=[
            LabelledBout(start=0, stop=3, name="rest"),
            LabelledBout(start=3, stop=6, name="swim"),
            LabelledBout(start=6, stop=8, name="turn"),  # the decoder has no turn
        ],
        _classes=ClassNames(names={0: "rest", 1: "swim"}, ignored=frozenset()),
        _summary=SimpleNamespace(setText=lambda text: None),
        _summary_text=lambda score: "",
        _fill_table=lambda score: None,
        _draw_matrix=lambda: None,
        _export_button=SimpleNamespace(setEnabled=lambda enabled: None),
        _status=SimpleNamespace(append=lambda text: None),
    )

    MetricsPanel._score_now(panel)

    [warning] = _messages(caplog, logging.WARNING)
    assert "classes ['turn'] are not among the class names ['rest', 'swim']" in warning
    assert _messages(caplog, logging.INFO) == [
        "Scored 2 bouts: accuracy 1.000, balanced accuracy 1.000 "
        "(0 ignored, 1 skipped)"
    ]


def test_scoring_before_every_input_is_loaded_logs_what_is_missing(caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    status = []
    panel = SimpleNamespace(
        _predictions=np.zeros(10, dtype=np.int64),
        _bouts=None,
        _classes=None,
        _status=SimpleNamespace(append=status.append),
    )

    MetricsPanel._on_score_clicked(panel)

    assert _messages(caplog, logging.WARNING) == [
        "Cannot score: load hand labels, class names first"
    ]
    assert status == ["Load hand labels, class names first."]
