"""Tests for poser._panels._factories."""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest

from poser._panels import _factories
from poser._panels.analysis_panel import AnalysisPanel
from poser._panels.annotation_panel import AnnotationPanel
from poser._panels.ethogram_panel import EthogramPanel
from poser.core import bout_detection
from poser.core.session import SessionManager
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
    resent = []
    panels = {
        "_data_cache": SimpleNamespace(active_entry_changed=_Signal()),
        "_analysis_cache": SimpleNamespace(bouts_changed=_Signal()),
        "_annotation_cache": SimpleNamespace(
            annotations_changed=_Signal(),
            labels_changed=_Signal(),
            show_entry=lambda entry: None,
            resend=lambda: resent.append(True),
            set_predictions=lambda predictions, checkpoint: None,
        ),
        "_ethogram_cache": SimpleNamespace(
            load_annotations=lambda labels: None,
            set_annotation_names=lambda names: None,
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
        "Connected the Annotation panel's label names to the Ethogram",
        "Connected the Inference panel's predictions to the Ethogram",
        "Connected the Inference panel's predictions to the Metrics panel",
        "Connected the Data panel's active entry to the Annotation panel",
        "Connected the Analysis panel's bouts to the Annotation panel",
        "Connected the Inference panel's predictions to the Annotation panel",
    ]
    assert panels["_inference_cache"].predictions_ready.slots[-1] is (
        panels["_annotation_cache"].set_predictions
    )
    assert panels["_data_cache"].active_entry_changed.slots == [
        panels["_annotation_cache"].show_entry
    ]
    assert panels["_analysis_cache"].bouts_changed.slots == [
        panels["_annotation_cache"].show_entry
    ]
    assert panels["_annotation_cache"].labels_changed.slots == [
        panels["_ethogram_cache"].set_annotation_names
    ]
    assert resent == [True]  # once, when the pair was connected


def test_the_panels_open_for_a_viewer_are_logged(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    viewer = object()
    for cache in (
        "_data_cache",
        "_analysis_cache",
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
    assert (
        "annotation=True ethogram=False inference=False metrics=False data=False "
        "analysis=False" in record.getMessage()
    )


def test_an_annotation_panel_opened_after_detection_lists_the_bouts(
    tmp_path, monkeypatch, qtbot
):
    # They were listed only after a click on Load Bouts.
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    session.activate(0)
    session.active.bouts = [
        {"start": 2, "end": 5, "label": 1},
        {"start": 10, "end": 12},
    ]
    viewer = SimpleNamespace(dims=SimpleNamespace(set_point=lambda axis, frame: None))
    monkeypatch.setattr(_factories.napari, "current_viewer", lambda: viewer)
    monkeypatch.setattr(_factories, "get_session", lambda viewer: session)
    monkeypatch.setattr(_factories, "_configure_logging", lambda: None)
    monkeypatch.setattr(_factories, "_enlarge_titlebar_buttons", lambda: None)
    monkeypatch.setattr(_factories, "_annotation_cache", {})
    monkeypatch.setattr(_factories, "_wired", set())

    panel = _factories.make_annotation_panel()
    qtbot.addWidget(panel)

    assert panel._bout_list.count() == 2
    assert panel._bout_list.currentRow() == 1  # the first unlabelled bout


def test_detected_bouts_are_listed_in_an_open_annotation_panel(
    tmp_path, monkeypatch, qtbot
):
    # They were listed only after a click on Load Bouts.
    found = [(2, 20), (30, 45)]
    monkeypatch.setattr(
        bout_detection,
        "orthogonal_variance",
        lambda points, center_node, fps, n_nodes: (found, None, None, None),
    )
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    session.activate(0)
    zeros = np.zeros((3, 50))  # (V, T)
    session.active.coords_data = {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}
    viewer = SimpleNamespace(dims=SimpleNamespace(set_point=lambda axis, frame: None))
    analysis = AnalysisPanel(viewer, session=session)
    annotation = AnnotationPanel(viewer, session=session)
    qtbot.addWidget(analysis)
    qtbot.addWidget(annotation)
    monkeypatch.setattr(_factories, "_analysis_cache", {id(viewer): analysis})
    monkeypatch.setattr(_factories, "_annotation_cache", {id(viewer): annotation})
    monkeypatch.setattr(_factories, "_wired", set())
    _factories._wire_panels(viewer)

    analysis._on_detect()

    assert annotation._bout_list.count() == 2
    assert annotation._bout_list.currentRow() == 0
    assert annotation._current_bouts is session.active.bouts


def test_an_ethogram_opened_after_labelling_shows_the_labels_by_name(
    tmp_path, monkeypatch, qtbot
):
    # It showed nothing until the next label click, then cls1 with an empty legend,
    # and its Annotations source still read None.
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    session.activate(0)
    zeros = np.zeros((3, 10))  # (V, T)
    session.active.coords_data = {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}
    session.active.bouts = [{"start": 2, "end": 5}]
    viewer = SimpleNamespace(
        dims=SimpleNamespace(
            current_step=(3,),
            set_point=lambda axis, frame: None,
            events=SimpleNamespace(current_step=_Signal()),
        )
    )
    annotation = AnnotationPanel(viewer, session=session, behaviour_labels={1: "swim"})
    qtbot.addWidget(annotation)
    annotation.show_entry(session.active)
    annotation._label_buttons_layout.itemAt(1).widget().click()  # [1] swim
    ethogram = EthogramPanel(viewer)
    qtbot.addWidget(ethogram)
    monkeypatch.setattr(_factories, "_annotation_cache", {id(viewer): annotation})
    monkeypatch.setattr(_factories, "_ethogram_cache", {id(viewer): ethogram})
    monkeypatch.setattr(_factories, "_wired", set())

    _factories._wire_panels(viewer)

    assert np.array_equal(ethogram._annotations, [0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    assert ethogram._ann_label.text() == "Annotation panel"
    status = re.sub(r"<[^>]+>|&nbsp;", " ", ethogram._frame_status.text()).split()
    assert status[:5] == ["Frame", "3", "|", "Ann:", "swim"]
    annotation._new_label_edit.setText("turn")
    annotation._on_add_label()
    legend = ethogram._ann_legend._lay
    names = [legend.itemAt(i).widget() for i in range(legend.count())]
    assert [w.text() for w in names if w is not None and w.text()] == [
        "0: other",
        "1: swim",
        "2: turn",
    ]
