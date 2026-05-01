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
from poser._panels.ethogram_panel import EthogramPanel
from poser.core.session import get_session

# Keep one panel instance per viewer so cross-panel signals can be wired
# regardless of which panel is opened first.
_ethogram_cache: dict = {}
_annotation_cache: dict = {}


def make_data_panel() -> DataPanel:
    v = napari.current_viewer()
    return DataPanel(v, session=get_session(v))


def make_annotation_panel() -> AnnotationPanel:
    v = napari.current_viewer()
    panel = AnnotationPanel(v, session=get_session(v))
    _annotation_cache[id(v)] = panel
    # Wire to ethogram if it is already open
    ethogram = _ethogram_cache.get(id(v))
    if ethogram is not None:
        panel.annotations_changed.connect(ethogram.load_annotations)
    return panel


def make_analysis_panel() -> AnalysisPanel:
    v = napari.current_viewer()
    return AnalysisPanel(v, session=get_session(v))


def make_inference_panel() -> InferencePanel:
    v = napari.current_viewer()
    panel = InferencePanel(v, session=get_session(v))
    # Wire predictions_ready → ethogram if one is already open
    ethogram = _ethogram_cache.get(id(v))
    if ethogram is not None:
        panel.predictions_ready.connect(
            lambda preds, ckpt: ethogram.load_predictions(preds, ckpt)
        )
    return panel


def make_train_panel() -> TrainPanel:
    v = napari.current_viewer()
    return TrainPanel(v, session=get_session(v))


def make_ethogram_panel() -> EthogramPanel:
    v = napari.current_viewer()
    panel = EthogramPanel(v)
    _ethogram_cache[id(v)] = panel
    # Wire to annotation panel if it is already open
    annotation = _annotation_cache.get(id(v))
    if annotation is not None:
        annotation.annotations_changed.connect(panel.load_annotations)
    return panel
