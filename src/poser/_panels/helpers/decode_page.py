"""Behaviour Decoding page of the Inference panel: load a decoder, label a file."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from napari.qt.threading import FunctionWorker, thread_worker
from qtpy.QtCore import Signal
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from poser._panels.helpers.option_help import help_button
from poser.core.inference import Inference
from poser.core.session import SessionManager
from poser.core.settings import resolve_device

log = logging.getLogger(__name__)


class DecodePage(QWidget):
    """Behaviour Decoding: load a decoder checkpoint, then label the active file."""

    predictions_ready = Signal(object, object)  # (predictions, checkpoint)

    def __init__(self, session: SessionManager) -> None:
        super().__init__()
        self._session = session
        self._inference = Inference()
        self._checkpoint: Path | None = None
        self._worker: FunctionWorker | None = None  # the job running now, if any
        self._build_ui()
        self._connect()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        checkpoint_group = QGroupBox("Decoder checkpoint")
        checkpoint_layout = QVBoxLayout(checkpoint_group)
        self._checkpoint_label = QLabel("No checkpoint loaded.")
        self._checkpoint_label.setWordWrap(True)
        checkpoint_layout.addWidget(self._checkpoint_label)
        self._settings_label = QLabel("")
        self._settings_label.setStyleSheet("font-size: 10px; color: grey;")
        checkpoint_layout.addWidget(self._settings_label)
        browse_row = QHBoxLayout()
        self._browse_button = QPushButton("Browse for checkpoint (.ckpt)…")
        browse_row.addWidget(self._browse_button)
        browse_row.addWidget(help_button("checkpoint"))
        checkpoint_layout.addLayout(browse_row)
        layout.addWidget(checkpoint_group)

        self._predict_button = QPushButton("▶  Predict behaviours (active file)")
        layout.addWidget(self._predict_button)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._status = QTextEdit()
        self._status.setReadOnly(True)
        self._status.setFixedHeight(120)
        # The platform's own fixed-width font. The generic name "monospace" does
        # not exist on macOS, so Qt scanned every installed font the first time
        # this page was shown, freezing the mode switch.
        self._status.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._status.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; }"
        )
        layout.addWidget(self._status)

    def _connect(self) -> None:
        self._browse_button.clicked.connect(self._on_browse_clicked)
        self._predict_button.clicked.connect(self._on_predict_clicked)

    def _on_browse_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select decoder checkpoint", "", "Checkpoints (*.ckpt);;All files (*)"
        )
        if not path:
            return
        path = Path(path)
        self._status.append(f"Loading decoder {path.name} …")
        worker = thread_worker(self._inference.model_load)(path, resolve_device())
        worker.returned.connect(lambda settings: self._on_model_loaded(path, settings))
        self._start(worker)

    def _on_model_loaded(self, path: Path, settings: dict) -> None:
        self._checkpoint = path
        self._checkpoint_label.setText(path.name)
        self._settings_label.setText(
            f"T2 {settings['T2']} · centre node {settings['center_node']} · "
            f"{settings['num_class']} classes"
        )
        self._status.append(f"Loaded decoder {path.name}.")

    def _on_predict_clicked(self) -> None:
        if self._checkpoint is None:
            self._status.append("Load a decoder checkpoint first.")
            return
        entry = self._session.active
        if entry is None:
            self._status.append("No active file. Activate one in the Data panel first.")
            return
        worker = thread_worker(self._inference.input_load)(
            entry.pose_path, entry.coords_data
        )
        worker.returned.connect(self._on_pose_loaded)
        self._start(worker)

    def _on_pose_loaded(self, form: str) -> None:
        self._status.append(f"Loaded pose as {form}.")
        self._status.append("Predicting is not wired up yet.")

    def _start(self, worker: FunctionWorker) -> None:
        """Run a background job, with the buttons off until it ends either way."""
        self._set_busy(True)
        worker.errored.connect(self._on_worker_errored)
        worker.finished.connect(lambda: self._set_busy(False))
        self._worker = worker
        worker.start()

    def _set_busy(self, busy: bool) -> None:
        self._browse_button.setEnabled(not busy)
        self._predict_button.setEnabled(not busy)
        # A range of 0 to 0 makes the bar show a moving "busy" pattern.
        self._progress.setRange(0, 0 if busy else 100)

    def _on_worker_errored(self, exc: Exception) -> None:
        log.error("Behaviour decoding failed", exc_info=exc)
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._status.append(f"ERROR: {exc}\n{details}")
