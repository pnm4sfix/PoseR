"""Tests for poser._panels._factories."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from poser._panels import _factories
from poser.core.settings import settings


@pytest.fixture
def poser_logger():
    """The poser logger, put back as it was so later tests still see its records."""
    logger = logging.getLogger("poser")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield logger
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def _handlers(logger: logging.Logger) -> dict:
    return {handler.get_name(): handler for handler in logger.handlers}


def test_debug_messages_go_to_the_log_file_in_data_logs(
    tmp_path, monkeypatch, poser_logger
):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")

    _factories._configure_logging()
    logging.getLogger("poser._panels.analysis_panel").debug("Detected %d bouts", 3)

    text = (tmp_path / "data" / "logs" / "poser.log").read_text()
    assert "DEBUG   poser._panels.analysis_panel: Detected 3 bouts" in text


def test_the_terminal_shows_the_configured_level_and_up(
    tmp_path, monkeypatch, poser_logger
):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "log_level", "WARNING")

    _factories._configure_logging()

    assert _handlers(poser_logger)["poser-terminal"].level == logging.WARNING
    assert _handlers(poser_logger)["poser-file"].level == logging.DEBUG


def test_each_message_is_handled_once(tmp_path, monkeypatch, poser_logger):
    # Every panel factory configures logging, and the napari launcher gives the
    # root logger a handler that prints whatever reaches it, DEBUG included.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")

    _factories._configure_logging()
    _factories._configure_logging()

    assert sorted(_handlers(poser_logger)) == ["poser-file", "poser-terminal"]
    assert len(poser_logger.handlers) == 2
    assert poser_logger.propagate is False


def test_an_unwritable_log_folder_leaves_terminal_logging(
    tmp_path, monkeypatch, poser_logger
):
    not_a_folder = tmp_path / "data"
    not_a_folder.write_text("a file where the data folder should be")
    monkeypatch.setattr(settings, "data_dir", not_a_folder)

    _factories._configure_logging()

    assert sorted(_handlers(poser_logger)) == ["poser-terminal"]


class _Signal:
    """Stands in for a Qt signal and records what was connected to it."""

    def __init__(self) -> None:
        self.slots = []

    def connect(self, slot) -> None:
        self.slots.append(slot)


def test_each_connection_between_panels_is_logged_once(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    viewer = object()
    panels = {
        "_annotation_cache": SimpleNamespace(annotations_changed=_Signal()),
        "_ethogram_cache": SimpleNamespace(
            load_annotations=lambda labels: None,
            load_predictions=lambda predictions, checkpoint: None,
        ),
        "_inference_cache": SimpleNamespace(predictions_ready=_Signal()),
        "_metrics_cache": SimpleNamespace(
            load_predictions=lambda predictions, checkpoint: None
        ),
    }
    for cache, panel in panels.items():
        monkeypatch.setattr(_factories, cache, {id(viewer): panel})
    monkeypatch.setattr(_factories, "_wired", set())

    _factories._wire_panels(viewer)
    _factories._wire_panels(viewer)

    connected = [
        record.getMessage()
        for record in caplog.records
        if record.name == "poser._panels._factories" and record.levelno == logging.INFO
    ]
    assert connected == [
        "Connected the Annotation panel's labels to the Ethogram",
        "Connected the Inference panel's predictions to the Ethogram",
        "Connected the Inference panel's predictions to the Metrics panel",
    ]


def test_the_panels_open_for_a_viewer_are_logged(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    viewer = object()
    for cache in (
        "_annotation_cache",
        "_ethogram_cache",
        "_inference_cache",
        "_metrics_cache",
    ):
        monkeypatch.setattr(_factories, cache, {})
    monkeypatch.setattr(
        _factories,
        "_annotation_cache",
        {id(viewer): SimpleNamespace(annotations_changed=_Signal())},
    )

    _factories._wire_panels(viewer)

    [record] = [r for r in caplog.records if r.name == "poser._panels._factories"]
    assert record.levelno == logging.DEBUG
    assert "annotation=True ethogram=False inference=False metrics=False" in (
        record.getMessage()
    )
