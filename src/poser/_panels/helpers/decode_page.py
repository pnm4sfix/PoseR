"""Behaviour Decoding page of the Inference panel: load a decoder, label files."""

from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from napari.qt.threading import FunctionWorker, GeneratorWorker, thread_worker
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
from poser.core.session import SessionEntry, SessionManager
from poser.core.settings import resolve_device

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _FileResult:
    """What happened to one pose file, sent from the worker thread to the page."""

    entry: SessionEntry
    prefix: str  # how status lines name this file, e.g. "[2/3] rec.h5"
    predictions: np.ndarray | None = None  # one label per frame
    saved_path: Path | None = None  # the .npy written, None if not saved
    error: Exception | None = None


def _traceback_text(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class DecodePage(QWidget):
    """Behaviour Decoding: load a decoder checkpoint, then label pose files."""

    predictions_ready = Signal(object, object)  # (predictions, checkpoint)

    def __init__(self, session: SessionManager) -> None:
        super().__init__()
        self._session = session
        self._inference = Inference()
        self._checkpoint: Path | None = None
        # The job running now, if any.
        self._worker: FunctionWorker | GeneratorWorker | None = None
        # The active file's latest predictions, which the Ethogram shows and
        # Export writes.
        self._shown: _FileResult | None = None
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
        self._predict_all_button = QPushButton(
            "▶▶  Predict behaviours (all session files)"
        )
        layout.addWidget(self._predict_all_button)

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

        self._export_button = QPushButton("Export predictions (JSON)…")
        self._export_button.setEnabled(False)
        layout.addWidget(self._export_button)

    def _connect(self) -> None:
        self._browse_button.clicked.connect(self._on_browse_clicked)
        self._predict_button.clicked.connect(self._on_predict_clicked)
        self._predict_all_button.clicked.connect(self._on_predict_all_clicked)
        self._export_button.clicked.connect(self._on_export_clicked)

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
        entry = self._session.active
        if entry is None:
            self._status.append("No active file. Activate one in the Data panel first.")
            return
        self._run([entry])

    def _on_predict_all_clicked(self) -> None:
        entries = [e for e in self._session.entries if e.pose_path or e.coords_data]
        if not entries:
            self._status.append(
                "No session file has pose data. Add some in the Data panel."
            )
            return
        self._run(entries)

    def _run(self, entries: list[SessionEntry]) -> None:
        if self._checkpoint is None:
            self._status.append("Load a decoder checkpoint first.")
            return
        worker = thread_worker(self._predict)(entries)
        worker.yielded.connect(self._on_worker_yielded)
        self._start(worker)

    def _predict(
        self, entries: list[SessionEntry]
    ) -> Iterator[int | str | _FileResult]:
        """Run in the worker thread, so it reads and writes no widgets.

        Yields status lines, the progress across all files as a percentage, and
        one _FileResult per file. A file that fails does not stop the rest.
        """
        n_failed = 0
        for index, entry in enumerate(entries):
            name = Path(entry.pose_path).name if entry.pose_path else "in-memory pose"
            prefix = name
            if len(entries) > 1:
                prefix = f"[{index + 1}/{len(entries)}] {name}"
            try:
                form = self._inference.input_load(entry.pose_path, entry.coords_data)
                n_frames = self._inference.n_frames
                yield f"{prefix}: loaded as {form}, {n_frames:,} frames. Predicting …"
                for message in self._inference.input_adapt():
                    yield f"{prefix}: Warning: {message}"

                batches, n_done = [], 0
                for labels in self._inference.iter_predictions():
                    batches.append(labels)
                    n_done += len(labels)
                    files_done = index + n_done / max(n_frames, 1)
                    yield int(100 * files_done / len(entries))
                predictions = np.concatenate(batches)

                saved = None
                try:
                    saved = self._inference.predictions_save(predictions)
                    yield f"{prefix}: saved {saved.name} and its .csv."
                except ValueError as exc:  # the pose came from memory, with no file
                    yield f"{prefix}: not saved: {exc}"
                yield _FileResult(
                    entry, prefix, predictions=predictions, saved_path=saved
                )
            except Exception as exc:
                n_failed += 1
                yield _FileResult(entry, prefix, error=exc)
            # Step on even when a file failed part way, so the bar reaches 100.
            yield int(100 * (index + 1) / len(entries))
        if len(entries) > 1:
            yield f"Finished {len(entries)} files, {n_failed} failed."

    def _on_worker_yielded(self, item: int | str | _FileResult) -> None:
        if isinstance(item, int):
            self._progress.setRange(0, 100)  # from the "busy" pattern to progress
            self._progress.setValue(item)
        elif isinstance(item, str):
            self._status.append(item)
        else:
            self._show_file_result(item)

    def _show_file_result(self, result: _FileResult) -> None:
        if result.error is not None:
            error = result.error
            log.error("Behaviour decoding failed on %s", result.prefix, exc_info=error)
            self._status.append(
                f"{result.prefix}: ERROR {error}\n{_traceback_text(error)}"
            )
            return
        labels, counts = np.unique(result.predictions, return_counts=True)
        summary = ", ".join(
            f"class {label}: {count:,}" for label, count in zip(labels, counts)
        )
        self._status.append(
            f"{result.prefix}: done, {len(result.predictions):,} frames. {summary}."
        )
        # The Ethogram and Metrics follow the file on screen, so a batch run
        # does not leave them showing whichever file happened to finish last.
        if result.entry is self._session.active:
            self._shown = result
            self.predictions_ready.emit(result.predictions, self._checkpoint)

    def _on_export_clicked(self) -> None:
        saved = self._shown.saved_path
        default = saved.with_suffix(".json") if saved else Path("predictions.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export predictions", str(default), "JSON (*.json)"
        )
        if not path:
            return
        frames = {str(f): int(label) for f, label in enumerate(self._shown.predictions)}
        try:
            Path(path).write_text(json.dumps(frames, indent=2))
        except OSError as exc:
            self._on_worker_errored(exc)
            return
        self._status.append(f"Exported {Path(path).name}.")

    def _start(self, worker: FunctionWorker | GeneratorWorker) -> None:
        """Run a background job, with the buttons off until it ends either way."""
        self._set_busy(True)
        worker.errored.connect(self._on_worker_errored)
        worker.finished.connect(lambda: self._set_busy(False))
        self._worker = worker
        worker.start()

    def _set_busy(self, busy: bool) -> None:
        self._browse_button.setEnabled(not busy)
        self._predict_button.setEnabled(not busy)
        self._predict_all_button.setEnabled(not busy)
        self._export_button.setEnabled(not busy and self._shown is not None)
        # A range of 0 to 0 makes the bar show a moving "busy" pattern.
        self._progress.setRange(0, 0 if busy else 100)

    def _on_worker_errored(self, exc: Exception) -> None:
        log.error("Behaviour decoding failed", exc_info=exc)
        self._status.append(f"ERROR: {exc}\n{_traceback_text(exc)}")
