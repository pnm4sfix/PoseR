"""Inference panel: pose estimation and behaviour decoding."""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget


import napari

from poser.core.session import SessionManager


class InferencePanel(QWidget):
    """Napari dock widget for pose estimation and behaviour decoding."""

    predictions_ready = Signal(object, object)  # (predictions, checkpoint)

    def __init__(self, viewer: napari.Viewer, session: SessionManager) -> None:
        super().__init__()
        self._viewer = viewer
        self._session = session
        self._build_ui()

    def _build_ui(self) -> None:
        QVBoxLayout(self)
