"""_panels/__init__.py — napari panel widgets."""
from poser._panels.data_panel import DataPanel
from poser._panels.annotation_panel import AnnotationPanel
from poser._panels.analysis_panel import AnalysisPanel
from poser._panels.inference_panel import InferencePanel
from poser._panels.train_panel import TrainPanel

__all__ = [
    "DataPanel",
    "AnnotationPanel",
    "AnalysisPanel",
    "InferencePanel",
    "TrainPanel",
]
