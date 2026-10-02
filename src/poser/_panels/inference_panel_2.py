"""Inference panel: pose estimation and behaviour decoding."""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


import napari

from poser._panels.helpers.decode_page import DecodePage
from poser._panels.helpers.pose_page import PosePage
from poser.core.session import SessionManager


class InferencePanel(QWidget):
    """Napari dock widget for pose estimation and behaviour decoding."""

    predictions_ready = Signal(object, object)  # (predictions, checkpoint)

    def __init__(self, viewer: napari.Viewer, session: SessionManager) -> None:
        super().__init__()
        self._viewer = viewer
        self._session = session
        self._build_ui()
        self._connect()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_group)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Pose Estimation", "Behaviour Decoding"])
        mode_layout.addWidget(self._mode_combo)
        layout.addWidget(mode_group)

        self._pose_page = PosePage(self._viewer, self._session)
        self._decode_page = DecodePage(self._session)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._pose_page)
        self._stack.addWidget(self._decode_page)
        layout.addWidget(self._stack)
        layout.addStretch()

    def _connect(self) -> None:
        self._mode_combo.currentIndexChanged.connect(self._stack.setCurrentIndex)
        self._decode_page.predictions_ready.connect(self.predictions_ready)
