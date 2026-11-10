"""Pose Estimation page of the Inference panel: run YOLO over session videos."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import napari
from napari.qt.threading import GeneratorWorker, thread_worker
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from poser._panels.helpers.option_help import help_button
from poser.core.io import save_coords_to_h5
from poser.core.pose_estimation import (
    POSER_PRETRAINED,
    YOLO_PRETRAINED,
    PoseEstimator,
)
from poser.core.schemas import InferenceMode
from poser.core.session import SessionEntry, SessionManager
from poser.core.settings import resolve_device

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _VideoResult:
    """What happened to one video, sent from the worker thread to the page."""

    index: int  # position in this run, from 0
    entry: SessionEntry
    pose_path: str = ""  # the saved coords file, empty when nothing was saved
    coords: dict | None = None
    error: Exception | None = None


def _option_row(label: str, widget: QWidget, help_key: str) -> QHBoxLayout:
    """One option: its label, its control and its "?" button."""
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    row.addWidget(widget)
    row.addWidget(help_button(help_key))
    row.addStretch()
    return row


def _traceback_text(exc: Exception) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class PosePage(QWidget):
    """Pose Estimation: run a YOLO pose model over the session's videos."""

    def __init__(self, viewer: napari.Viewer, session: SessionManager) -> None:
        super().__init__()
        self._viewer = viewer
        self._session = session
        self._estimator = PoseEstimator()
        self._worker: GeneratorWorker | None = None  # the job running now, if any
        self._n_videos = 0  # videos in the current run, for "[1/3]" messages
        self._build_ui()
        self._connect()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_options_group())

        self._run_active_button = QPushButton("▶  Run pose estimation (active video)")
        layout.addWidget(self._run_active_button)
        self._run_all_button = QPushButton(
            "▶▶  Run pose estimation (all session videos)"
        )
        layout.addWidget(self._run_all_button)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)

        self._status = QTextEdit()
        self._status.setReadOnly(True)
        self._status.setFixedHeight(120)
        # The platform's own fixed-width font. The generic name "monospace" makes
        # Qt scan every installed font on macOS the first time it is shown.
        self._status.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._status.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; }"
        )
        layout.addWidget(self._status)

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Pose model")
        layout = QVBoxLayout(group)
        self._model_combo = QComboBox()
        self._model_combo.addItems([*POSER_PRETRAINED, *YOLO_PRETRAINED])
        layout.addLayout(
            _option_row("Pretrained model:", self._model_combo, "pretrained_model")
        )
        hint = QLabel(
            "zeb / fly3 / mouse7 / mouse13 are PoseR species models, downloaded "
            "from GitHub on first use."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: grey;")
        layout.addWidget(hint)
        self._browse_button = QPushButton("Browse for custom .pt…")
        layout.addWidget(self._browse_button)
        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Options")
        layout = QVBoxLayout(group)

        self._individuals_spin = QSpinBox()
        self._individuals_spin.setRange(1, 100)
        layout.addLayout(
            _option_row("Max individuals:", self._individuals_spin, "max_individuals")
        )

        self._mode_combo = QComboBox()
        self._mode_combo.addItem(
            "predict  (fast / no tracking)", InferenceMode.PREDICT.value
        )
        self._mode_combo.addItem(
            "track  (sequential / with tracking)", InferenceMode.TRACK.value
        )
        layout.addLayout(
            _option_row("Inference mode:", self._mode_combo, "inference_mode")
        )

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 256)
        self._batch_spin.setValue(16)
        layout.addLayout(
            _option_row("Batch size (predict only):", self._batch_spin, "batch_size")
        )

        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(0, 8192)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setSpecialValueText("auto")  # shown for 0
        layout.addLayout(_option_row("Image size:", self._imgsz_spin, "image_size"))

        self._camera_spin = QSpinBox()
        self._camera_spin.setRange(0, 31)
        layout.addLayout(
            _option_row("Zarr camera axis:", self._camera_spin, "zarr_camera")
        )
        return group

    def _connect(self) -> None:
        self._browse_button.clicked.connect(self._on_browse_clicked)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._run_active_button.clicked.connect(self._on_run_active_clicked)
        self._run_all_button.clicked.connect(self._on_run_all_clicked)

    def _mode(self) -> InferenceMode:
        return InferenceMode(self._mode_combo.currentData())

    def _on_browse_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select pose model", "", "YOLO weights (*.pt);;All files (*)"
        )
        if path:
            self._model_combo.insertItem(0, path)
            self._model_combo.setCurrentIndex(0)

    def _on_mode_changed(self) -> None:
        # Track runs one frame at a time, so batch size does not apply.
        self._batch_spin.setEnabled(self._mode() is InferenceMode.PREDICT)

    def _on_run_active_clicked(self) -> None:
        entry = self._session.active
        if entry is None or not entry.video_path:
            self._status.append(
                "No active video. Add one in the Data panel and activate it first."
            )
            return
        self._run([entry])

    def _on_run_all_clicked(self) -> None:
        entries = [entry for entry in self._session.entries if entry.video_path]
        if not entries:
            self._status.append("No session entries have a video.")
            return
        self._run(entries)

    def _run(self, entries: list[SessionEntry]) -> None:
        """Read every option here, on the UI thread, then start the worker."""
        model = self._model_combo.currentText()
        camera = self._camera_spin.value()
        options = {
            "mode": self._mode(),
            "batch_size": self._batch_spin.value(),
            "max_individuals": self._individuals_spin.value(),
            "imgsz": self._imgsz_spin.value() or None,  # 0 means auto
        }
        self._n_videos = len(entries)
        self._progress.setValue(0)
        self._status.append(f"Loading {Path(model).name} …")

        worker = thread_worker(self._estimate_videos)(entries, model, camera, options)
        worker.yielded.connect(self._on_worker_yielded)
        worker.returned.connect(lambda: self._status.append("Pose estimation done."))
        worker.errored.connect(self._on_worker_errored)
        worker.finished.connect(lambda: self._set_busy(False))
        self._set_busy(True)
        self._worker = worker
        worker.start()

    def _estimate_videos(
        self,
        entries: list[SessionEntry],
        model: str,
        camera: int,
        options: dict,
    ) -> Iterator[int | str | _VideoResult]:
        """Run in the worker thread, so it reads and writes no widgets.

        Yields the run's progress as a percentage, status lines, and one
        _VideoResult per video.
        """
        self._estimator.model_load(Path(model), resolve_device())
        yield f"Loaded {Path(model).name}."
        for index, entry in enumerate(entries):
            name = Path(entry.video_path).name
            try:
                n_frames = self._estimator.input_load(entry.video_path, camera)
                yield f"[{index + 1}/{len(entries)}] {name}: {n_frames:,} frames …"
                records = []
                for keypoints in self._estimator.iter_keypoints(**options):
                    records.append(keypoints)
                    videos_done = index + (keypoints.frame + 1) / max(n_frames, 1)
                    yield min(100, int(100 * videos_done / len(entries)))
                coords = self._estimator.coords_from_keypoints(records)
                pose_path = ""
                if coords:
                    pose_path = save_coords_to_h5(coords, entry.video_path)
                yield _VideoResult(index, entry, pose_path=pose_path, coords=coords)
            except Exception as exc:
                # One bad video must not stop the rest of the run (FR-P4).
                yield _VideoResult(index, entry, error=exc)

    def _on_worker_yielded(self, item: int | str | _VideoResult) -> None:
        if isinstance(item, int):
            self._progress.setValue(item)
        elif isinstance(item, str):
            self._status.append(item)
        else:
            self._show_result(item)

    def _show_result(self, result: _VideoResult) -> None:
        name = Path(result.entry.video_path).name
        prefix = f"[{result.index + 1}/{self._n_videos}] {name}"
        if result.error is not None:
            log.error("Pose estimation failed on %s", name, exc_info=result.error)
            self._status.append(
                f"{prefix}: ERROR {result.error}\n{_traceback_text(result.error)}"
            )
            return
        if not result.coords:
            self._status.append(f"{prefix}: no animals found, nothing saved.")
            return
        self._add_points_layer(result.entry, result.pose_path, result.coords)
        self._status.append(
            f"{prefix}: saved {Path(result.pose_path).name}, "
            f"{len(result.coords)} individual(s)."
        )

    def _add_points_layer(
        self, entry: SessionEntry, pose_path: str, coords: dict
    ) -> None:
        """Link the new pose file to its entry and draw it, replacing a rerun's."""
        entry.pose_path = pose_path
        entry.coords_data = coords
        name = entry.layer_name("points")
        if name in self._viewer.layers:
            self._viewer.layers.remove(self._viewer.layers[name])
        points, properties = PoseEstimator.points_from_coords(coords)
        self._viewer.add_points(
            points,
            properties=properties,
            name=name,
            size=3,
            opacity=0.8,
            out_of_slice_display=False,
        )
        if name not in entry.layer_names:
            entry.layer_names.append(name)

    def _set_busy(self, busy: bool) -> None:
        self._browse_button.setEnabled(not busy)
        self._run_active_button.setEnabled(not busy)
        self._run_all_button.setEnabled(not busy)

    def _on_worker_errored(self, exc: Exception) -> None:
        log.error("Pose estimation failed", exc_info=exc)
        self._status.append(f"ERROR: {exc}\n{_traceback_text(exc)}")
