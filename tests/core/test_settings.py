"""Tests for poser.core.settings."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from poser.core import settings as settings_module
from poser.core.settings import (
    Settings,
    _find_data_dir,
    resolve_device,
    resolve_output_dir,
    settings,
)


def test_source_checkout_keeps_data_in_the_repository(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")

    assert _find_data_dir(tmp_path / "src" / "poser") == tmp_path / "data"


def test_installed_package_keeps_data_in_the_home_folder(tmp_path):
    # An installed poser sits in site-packages, with no repository around it.
    package_dir = tmp_path / "lib" / "site-packages" / "poser"

    assert _find_data_dir(package_dir) == Path.home() / ".poser" / "data"


def test_data_dir_can_be_set_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("POSER_DATA_DIR", str(tmp_path / "elsewhere"))

    assert Settings().data_dir == tmp_path / "elsewhere"


def test_output_dir_is_a_data_dir_subfolder_made_on_first_use(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")

    folder = resolve_output_dir("pose")

    assert folder == tmp_path / "data" / "pose"
    assert folder.is_dir()


def test_log_level_is_read_from_the_environment_in_any_case(monkeypatch):
    monkeypatch.setenv("POSER_LOG_LEVEL", "debug")

    assert Settings().log_level == "DEBUG"


def test_unknown_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("POSER_LOG_LEVEL", "verbose")

    with pytest.raises(ValidationError, match="log_level"):
        Settings()


def test_a_named_device_is_logged(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    monkeypatch.setattr(settings, "device", "cpu")

    resolve_device()

    [record] = [r for r in caplog.records if r.name == "poser.core.settings"]
    assert record.levelno == logging.INFO
    assert record.getMessage() == "Using device cpu, as POSER_DEVICE asks"


def test_the_device_auto_picks_is_logged_with_the_reason(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    monkeypatch.setattr(settings, "device", "auto")
    torch = settings_module.torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    resolve_device()

    [record] = [r for r in caplog.records if r.name == "poser.core.settings"]
    assert record.getMessage() == "Using device cpu: neither CUDA nor MPS is available"
