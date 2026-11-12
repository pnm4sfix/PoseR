"""Tests for poser.core.settings."""

from __future__ import annotations

from pathlib import Path

from poser.core.settings import Settings, _find_data_dir, resolve_output_dir, settings


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
