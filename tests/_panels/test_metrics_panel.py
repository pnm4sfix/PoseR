"""Tests for poser._panels.metrics_panel."""

from __future__ import annotations

from types import SimpleNamespace

from poser._panels import metrics_panel
from poser._panels.metrics_panel import MetricsPanel
from poser.core.settings import settings

from .conftest import Dialogs


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
