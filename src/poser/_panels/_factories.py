"""Thin factory functions used as napari widget entry points.

Using napari.current_viewer() avoids dependency injection entirely: napari
guarantees a viewer exists before any widget command can be triggered.
"""

import napari
from qtpy.QtWidgets import QApplication

from poser._panels.data_panel import DataPanel
from poser._panels.annotation_panel import AnnotationPanel
from poser._panels.analysis_panel import AnalysisPanel
from poser._panels.inference_panel_2 import InferencePanel
from poser._panels.train_panel import TrainPanel
from poser._panels.ethogram_panel import EthogramPanel
from poser.core.session import get_session

# One instance of each wirable panel per viewer. Panels talk to each other
# through Qt signals, which SessionManager cannot carry because core must stay
# free of Qt (STYLEGUIDE 1.2), so the connections are made out here instead.
_annotation_cache: dict = {}
_ethogram_cache: dict = {}
_inference_cache: dict = {}

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


def make_data_panel() -> DataPanel:
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    return DataPanel(v, session=get_session(v))


def make_annotation_panel() -> AnnotationPanel:
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    panel = AnnotationPanel(v, session=get_session(v))
    _annotation_cache[id(v)] = panel
    _wire_panels(v)
    return panel


def make_analysis_panel() -> AnalysisPanel:
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    return AnalysisPanel(v, session=get_session(v))


def make_inference_panel() -> InferencePanel:
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    panel = InferencePanel(v, session=get_session(v))
    _inference_cache[id(v)] = panel
    _wire_panels(v)
    return panel


def make_train_panel() -> TrainPanel:
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    return TrainPanel(v, session=get_session(v))


def make_ethogram_panel() -> EthogramPanel:
    _enlarge_titlebar_buttons()
    v = napari.current_viewer()
    panel = EthogramPanel(v)
    _ethogram_cache[id(v)] = panel
    _wire_panels(v)
    return panel
