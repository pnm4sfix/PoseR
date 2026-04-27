"""
_panels/_factories.py
~~~~~~~~~~~~~~~~~~~~~
Thin factory functions used as napari widget entry points.

Using ``napari.current_viewer()`` avoids dependency injection entirely —
napari guarantees a viewer exists before any widget command can be triggered.
"""
import napari

from poser._panels.data_panel import DataPanel
from poser._panels.annotation_panel import AnnotationPanel
from poser._panels.analysis_panel import AnalysisPanel
from poser._panels.inference_panel import InferencePanel
from poser._panels.train_panel import TrainPanel
from poser.core.session import get_session


def make_data_panel() -> DataPanel:
    v = napari.current_viewer()
    return DataPanel(v, session=get_session(v))


def make_annotation_panel() -> AnnotationPanel:
    v = napari.current_viewer()
    return AnnotationPanel(v, session=get_session(v))


def make_analysis_panel() -> AnalysisPanel:
    v = napari.current_viewer()
    return AnalysisPanel(v, session=get_session(v))


def make_inference_panel() -> InferencePanel:
    v = napari.current_viewer()
    return InferencePanel(v, session=get_session(v))


def make_train_panel() -> TrainPanel:
    v = napari.current_viewer()
    return TrainPanel(v, session=get_session(v))
