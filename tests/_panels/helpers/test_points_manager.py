"""Tests for poser._panels.helpers.points_manager."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from poser._panels.helpers.points_manager import PointsLayerManager


class _Viewer:
    """Stands in for napari.Viewer, keeping its points layers in a plain list."""

    def __init__(self) -> None:
        self.layers: list[SimpleNamespace] = []

    def add_points(self, data, name, **style) -> SimpleNamespace:
        layer = SimpleNamespace(data=data, name=name)
        self.layers.append(layer)
        return layer


def _records(caplog) -> list[logging.LogRecord]:
    """What points_manager logged, in order."""
    return [
        record
        for record in caplog.records
        if record.name == "poser._panels.helpers.points_manager"
    ]


def test_adding_an_individual_logs_its_new_layer(caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    manager = PointsLayerManager(_Viewer())

    manager.add_individual("ind1", n_nodes=9)

    [added] = [record for record in _records(caplog) if record.levelno == logging.INFO]
    assert "pose_ind1" in added.getMessage()
    assert "9 nodes" in added.getMessage()


@pytest.mark.parametrize("method", ["flush", "clear_buffer"])
def test_flushing_or_clearing_an_unknown_individual_logs_a_warning(caplog, method):
    caplog.set_level(logging.DEBUG, logger="poser")
    manager = PointsLayerManager(_Viewer())
    manager.add_individual("ind1", n_nodes=9)

    getattr(manager, method)("ind2")

    [warning] = [
        record for record in _records(caplog) if record.levelno == logging.WARNING
    ]
    assert "ind2" in warning.getMessage()
    assert "no layer registered" in warning.getMessage()


def test_a_layer_already_deleted_in_napari_is_logged_with_the_error(caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    viewer = _Viewer()
    manager = PointsLayerManager(viewer)
    manager.add_individual("ind1", n_nodes=9)
    viewer.layers.clear()  # the user deleted the layer from napari's layer list

    manager.remove_individual("ind1")

    [failed] = [record for record in _records(caplog) if record.exc_info]
    assert failed.levelno == logging.DEBUG
    assert "ind1" in failed.getMessage()
    assert manager.keys() == []
