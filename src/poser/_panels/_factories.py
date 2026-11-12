"""Thin factory functions used as napari widget entry points.

Using napari.current_viewer() avoids dependency injection entirely: napari
guarantees a viewer exists before any widget command can be triggered.
"""

import logging
from logging.handlers import RotatingFileHandler

import napari
from qtpy.QtWidgets import QApplication

from poser._panels.data_panel import DataPanel
from poser._panels.annotation_panel import AnnotationPanel
from poser._panels.analysis_panel import AnalysisPanel
from poser._panels.inference_panel import InferencePanel
from poser._panels.train_panel import TrainPanel
from poser._panels.ethogram_panel import EthogramPanel
from poser._panels.metrics_panel import MetricsPanel
from poser.core.session import get_session
from poser.core.settings import resolve_output_dir, settings

log = logging.getLogger(__name__)

# One instance of each wirable panel per viewer. Panels talk to each other
# through Qt signals, which SessionManager cannot carry because core must stay
# free of Qt (STYLEGUIDE 1.2), so the connections are made out here instead.
_annotation_cache: dict = {}
_ethogram_cache: dict = {}
_inference_cache: dict = {}
_metrics_cache: dict = {}

# Pairs already connected, keyed by the two panel objects. Panels are rewired
# on every factory call so open order cannot matter, and this stops a signal
# being connected twice and firing twice.
_wired: set = set()

# napari styles the dock title-bar buttons at 12x12 px, which is an awkward
# target. Scaling them up is cosmetic and applies to every dock in the window,
# PoseR's and napari's alike.
_TITLEBAR_BUTTON_SIZE_PX = 22
_TITLEBAR_QSS = f"""
#QTitleBarCloseButton, #QTitleBarFloatButton, #QTitleBarHideButton {{
    width: {_TITLEBAR_BUTTON_SIZE_PX}px;
    height: {_TITLEBAR_BUTTON_SIZE_PX}px;
}}
"""


# The terminal shows settings.log_level and up; the file keeps everything.
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_LOG_FILE_BYTES = 5 * 1024 * 1024
_LOG_FILE_BACKUPS = 3


def _configure_logging() -> None:
    """Send poser's log messages to the terminal and to data/logs/poser.log, once.

    The poser logger stops passing messages up to the root logger: the napari
    launcher gives that one a handler which prints everything it receives, so
    every message would appear twice, DEBUG included.
    """
    logger = logging.getLogger("poser")
    if any(handler.get_name() == "poser-terminal" for handler in logger.handlers):
        return
    formatter = logging.Formatter(_LOG_FORMAT)
    terminal = logging.StreamHandler()
    terminal.set_name("poser-terminal")
    terminal.setLevel(settings.log_level)
    terminal.setFormatter(formatter)
    logger.addHandler(terminal)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    try:
        log_file = resolve_output_dir("logs") / "poser.log"
        to_file = RotatingFileHandler(
            log_file,
            maxBytes=_LOG_FILE_BYTES,
            backupCount=_LOG_FILE_BACKUPS,
            encoding="utf-8",
        )
    except OSError:
        log.warning(
            "Cannot write a log file under %s; logging to the terminal only",
            settings.data_dir,
            exc_info=True,
        )
        return
    to_file.set_name("poser-file")
    to_file.setLevel(logging.DEBUG)
    to_file.setFormatter(formatter)
    logger.addHandler(to_file)
    log.info(
        "Logging to %s; the terminal shows %s and up", log_file, settings.log_level
    )


def _enlarge_titlebar_buttons() -> None:
    """Scale up the dock title-bar buttons, once per application."""
    app = QApplication.instance()
    if app is None or _TITLEBAR_QSS in app.styleSheet():
        return
    app.setStyleSheet(app.styleSheet() + _TITLEBAR_QSS)


def _wire_panels(viewer) -> None:
    """Connect every pair of panels currently open for this viewer.

    Called after each panel is built, so a pair is connected as soon as its
    second half appears. Wiring only at construction meant a panel opened
    later was never connected, and predictions or annotations silently never
    reached the ethogram.
    """
    key = id(viewer)
    annotation = _annotation_cache.get(key)
    ethogram = _ethogram_cache.get(key)
    inference = _inference_cache.get(key)
    metrics = _metrics_cache.get(key)

    if annotation is not None and ethogram is not None:
        pair = (id(annotation), id(ethogram), "annotations")
        if pair not in _wired:
            annotation.annotations_changed.connect(ethogram.load_annotations)
            _wired.add(pair)

    if inference is not None and ethogram is not None:
        pair = (id(inference), id(ethogram), "predictions")
        if pair not in _wired:
            inference.predictions_ready.connect(ethogram.load_predictions)
            _wired.add(pair)

    if inference is not None and metrics is not None:
        pair = (id(inference), id(metrics), "metrics_predictions")
        if pair not in _wired:
            inference.predictions_ready.connect(metrics.load_predictions)
            _wired.add(pair)


def make_data_panel() -> DataPanel:
    _configure_logging()
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    return DataPanel(v, session=get_session(v))


def make_annotation_panel() -> AnnotationPanel:
    _configure_logging()
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    panel = AnnotationPanel(v, session=get_session(v))
    _annotation_cache[id(v)] = panel
    _wire_panels(v)
    return panel


def make_analysis_panel() -> AnalysisPanel:
    _configure_logging()
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    return AnalysisPanel(v, session=get_session(v))


def make_inference_panel() -> InferencePanel:
    _configure_logging()
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    panel = InferencePanel(v, session=get_session(v))
    _inference_cache[id(v)] = panel
    _wire_panels(v)
    return panel


def make_train_panel() -> TrainPanel:
    _configure_logging()
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    return TrainPanel(v, session=get_session(v))


def make_ethogram_panel() -> EthogramPanel:
    _configure_logging()
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    panel = EthogramPanel(v)
    _ethogram_cache[id(v)] = panel
    _wire_panels(v)
    return panel


def make_metrics_panel() -> MetricsPanel:
    _configure_logging()
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    panel = MetricsPanel(v, session=get_session(v))
    _metrics_cache[id(v)] = panel
    _wire_panels(v)
    return panel
