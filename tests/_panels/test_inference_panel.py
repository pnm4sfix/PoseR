"""Tests for poser._panels.inference_panel."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from poser._panels.inference_panel import InferencePanel


class _Signal:
    """Stands in for a Qt signal and records what was connected to it."""

    def __init__(self) -> None:
        self.slots = []

    def connect(self, slot) -> None:
        self.slots.append(slot)


def _panel() -> SimpleNamespace:
    """A stand-in panel showing Pose Estimation, whose signals record connections."""
    return SimpleNamespace(
        _mode_combo=SimpleNamespace(
            currentIndexChanged=_Signal(),
            currentTextChanged=_Signal(),
            currentText=lambda: "Pose Estimation",
        ),
        _stack=SimpleNamespace(setCurrentIndex=lambda index: None),
        _decode_page=SimpleNamespace(predictions_ready=_Signal()),
        predictions_ready=_Signal(),
    )


def _records(caplog) -> list[logging.LogRecord]:
    """What inference_panel logged, in order."""
    return [
        record
        for record in caplog.records
        if record.name == "poser._panels.inference_panel"
    ]


def test_wiring_logs_that_decoded_predictions_are_relayed(caplog):
    # Predictions reach the Ethogram only through this relay, so the log shows it.
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = _panel()

    InferencePanel._connect(panel)

    assert panel._decode_page.predictions_ready.slots == [panel.predictions_ready]
    [record] = _records(caplog)
    assert record.levelno == logging.DEBUG
    assert "predictions are relayed as predictions_ready" in record.getMessage()


def test_wiring_logs_the_mode_the_panel_starts_in(caplog):
    caplog.set_level(logging.DEBUG, logger="poser")

    InferencePanel._connect(_panel())

    [record] = _records(caplog)
    assert "starting in Pose Estimation mode" in record.getMessage()


def test_switching_page_is_logged(caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = _panel()
    InferencePanel._connect(panel)
    [on_switch] = panel._mode_combo.currentTextChanged.slots

    on_switch("Behaviour Decoding")

    assert _records(caplog)[-1].getMessage() == (
        "Inference panel switched to Behaviour Decoding"
    )
