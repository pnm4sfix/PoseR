"""Tests for poser._panels._factories."""

from __future__ import annotations

import logging

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
