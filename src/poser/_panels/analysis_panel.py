"""
_panels/analysis_panel.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Analysis Panel — bout detection and manual bout definition.

Features:
* Automatic bout detection via orthogonal or egocentric variance
* Threshold slider with live preview on a matplotlib figure
* Manual bout definition: user adjusts the frame slider and clicks
  Start / End to define bout boundaries (Feature 5)
* Frame export for finetuning (Feature 3)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from poser.core.bout_detection import manual_bout
from poser.core.session import SessionManager

if TYPE_CHECKING:
    import napari


class AnalysisPanel(QWidget):
    """Bout detection and manual annotation widget.

    Parameters
    ----------
    viewer:
        The running ``napari.Viewer`` instance.
    session:
        Shared :class:`~poser.core.session.SessionManager`.
    on_bouts_updated:
        Optional callback ``(bouts: list) -> None`` called whenever the
        bout list changes (e.g. to populate the AnnotationPanel).
    """

    def __init__(
        self,
        viewer: "napari.Viewer",
        session: SessionManager,
        on_bouts_updated: Optional[Callable[[List[dict]], None]] = None,
    ):
        super().__init__()
        self._viewer = viewer
        self._session = session
        self._on_bouts_updated = on_bouts_updated
        self._detected_bouts: List[dict] = []
        self._manual_start: Optional[int] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Parameters ────────────────────────────────────────────────
        params_grp = QGroupBox("Bout Detection Parameters")
        params_lay = QVBoxLayout(params_grp)

        # Method selector
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["orthogonal", "egocentric"])
        method_row.addWidget(self._method_combo)
        params_lay.addLayout(method_row)

        # FPS
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 500)
        self._fps_spin.setValue(25)
        fps_row.addWidget(self._fps_spin)
        params_lay.addLayout(fps_row)

        # Individual
        ind_row = QHBoxLayout()
        ind_row.addWidget(QLabel("Individual:"))
        self._individual_combo = QComboBox()
        self._individual_combo.addItem("ind1")
        ind_row.addWidget(self._individual_combo)
        params_lay.addLayout(ind_row)

        # Threshold
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold:"))
        self._thresh_spin = QDoubleSpinBox()
        self._thresh_spin.setRange(0.0, 1000.0)
        self._thresh_spin.setSingleStep(0.5)
        self._thresh_spin.setValue(5.0)
        self._thresh_spin.setDecimals(2)
        thresh_row.addWidget(self._thresh_spin)
        params_lay.addLayout(thresh_row)

        # Min / max bout length
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Min frames:"))
        self._min_frames_spin = QSpinBox()
        self._min_frames_spin.setRange(1, 10000)
        self._min_frames_spin.setValue(10)
        dur_row.addWidget(self._min_frames_spin)
        dur_row.addWidget(QLabel("Max frames:"))
        self._max_frames_spin = QSpinBox()
        self._max_frames_spin.setRange(1, 100000)
        self._max_frames_spin.setValue(300)
        dur_row.addWidget(self._max_frames_spin)
        params_lay.addLayout(dur_row)

        root.addWidget(params_grp)

        # ── Run auto detection ────────────────────────────────────────
        btn_detect = QPushButton("▶  Detect Bouts Automatically")
        btn_detect.clicked.connect(self._on_detect)
        root.addWidget(btn_detect)

        self._bout_count_label = QLabel("Bouts detected: —")
        root.addWidget(self._bout_count_label)

        # ── Manual bout definition  (Feature 5) ──────────────────────
        manual_grp = QGroupBox("Manual Bout Definition")
        manual_lay = QVBoxLayout(manual_grp)

        manual_info = QLabel(
            "Adjust the viewer frame slider, then click [Set Start] / [Set End]."
        )
        manual_info.setWordWrap(True)
        manual_info.setStyleSheet("font-size: 10px; color: grey;")
        manual_lay.addWidget(manual_info)

        # Current frame readout
        self._frame_label = QLabel("Current frame: 0")
        manual_lay.addWidget(self._frame_label)

        # Sync with viewer dims
        try:
            self._viewer.dims.events.current_step.connect(self._on_frame_changed)
        except Exception:
            pass

        btn_row = QHBoxLayout()
        self._btn_set_start = QPushButton("Set Start")
        self._btn_set_start.clicked.connect(self._on_set_start)
        self._btn_set_end = QPushButton("Set End  →  Add Bout")
        self._btn_set_end.clicked.connect(self._on_set_end)
        self._btn_set_end.setEnabled(False)
        btn_row.addWidget(self._btn_set_start)
        btn_row.addWidget(self._btn_set_end)
        manual_lay.addLayout(btn_row)

        self._manual_range_label = QLabel("Range: — to —")
        manual_lay.addWidget(self._manual_range_label)

        root.addWidget(manual_grp)

        # ── Frame export  (Feature 3) ─────────────────────────────────
        export_grp = QGroupBox("Export Frame for Finetuning")
        export_lay = QVBoxLayout(export_grp)

        export_type_row = QHBoxLayout()
        export_type_row.addWidget(QLabel("Format:"))
        self._export_type_combo = QComboBox()
        self._export_type_combo.addItems(["YOLO (pose estimation)", "Classification (behaviour)"])
        export_type_row.addWidget(self._export_type_combo)
        export_lay.addLayout(export_type_row)

        btn_export_frame = QPushButton("Save Current Frame…")
        btn_export_frame.clicked.connect(self._on_export_frame)
        export_lay.addWidget(btn_export_frame)

        root.addWidget(export_grp)
        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_frame_changed(self, event=None) -> None:
        try:
            frame = int(self._viewer.dims.current_step[0])
            self._frame_label.setText(f"Current frame: {frame}")
        except Exception:
            pass

    def _on_set_start(self) -> None:
        try:
            self._manual_start = int(self._viewer.dims.current_step[0])
        except Exception:
            self._manual_start = 0
        self._manual_range_label.setText(f"Range: {self._manual_start} to —")
        self._btn_set_end.setEnabled(True)

    def _on_set_end(self) -> None:
        try:
            end_frame = int(self._viewer.dims.current_step[0])
        except Exception:
            end_frame = self._manual_start or 0

        if self._manual_start is None or end_frame <= self._manual_start:
            QMessageBox.warning(
                self, "Invalid range",
                "End frame must be after start frame."
            )
            return

        # Build a manual bout using the active session entry's coords_data
        entry = self._session.active_entry
        coords_data = entry.coords_data if entry else None
        individual = self._individual_combo.currentText()

        bout = manual_bout(
            start=self._manual_start,
            end=end_frame,
            coords_data=coords_data,
            individual_key=individual,
        )
        self._detected_bouts.append(bout)
        self._update_bout_count()
        if self._on_bouts_updated:
            self._on_bouts_updated(list(self._detected_bouts))

        self._manual_range_label.setText(
            f"Added: {self._manual_start} → {end_frame}  "
            f"(total {len(self._detected_bouts)} bouts)"
        )
        self._manual_start = None
        self._btn_set_end.setEnabled(False)

    def _on_detect(self) -> None:
        entry = self._session.active_entry
        if entry is None or entry.coords_data is None:
            QMessageBox.warning(self, "No data", "Load a pose file first.")
            return

        from poser.core.bout_detection import orthogonal_variance, egocentric_variance

        individual = self._individual_combo.currentText()
        coords_data = entry.coords_data

        if individual not in coords_data:
            available = list(coords_data.keys())
            QMessageBox.warning(
                self, "Individual not found",
                f"'{individual}' not in data.  Available: {available}"
            )
            return

        ind = coords_data[individual]
        x = np.array(ind["x"])
        y = np.array(ind["y"])
        points = np.stack([x, y], axis=-1)  # (T, V, 2)

        fps = self._fps_spin.value()
        method = self._method_combo.currentText()
        n_nodes = points.shape[1] if points.ndim == 3 else 9

        try:
            if method == "egocentric":
                bouts, *_ = egocentric_variance(points, fps=fps, n_nodes=n_nodes)
            else:
                bouts, *_ = orthogonal_variance(points, fps=fps, n_nodes=n_nodes)
        except Exception as exc:
            QMessageBox.critical(self, "Detection error", str(exc))
            return

        # Filter by duration
        min_f = self._min_frames_spin.value()
        max_f = self._max_frames_spin.value()
        bouts = [b for b in bouts if min_f <= (b["end"] - b["start"]) <= max_f]

        self._detected_bouts = bouts
        self._update_bout_count()
        if self._on_bouts_updated:
            self._on_bouts_updated(list(bouts))

    def _on_export_frame(self) -> None:
        try:
            frame_idx = int(self._viewer.dims.current_step[0])
        except Exception:
            frame_idx = 0

        entry = self._session.active_entry
        export_type = self._export_type_combo.currentText()

        output_dir = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not output_dir:
            return

        individual = self._individual_combo.currentText()

        if "YOLO" in export_type:
            # Need a frame image — try to get from viewer
            frame_image = self._get_frame_image(frame_idx)
            if frame_image is None:
                QMessageBox.warning(self, "No image", "No image layer found in viewer.")
                return
            keypoints = self._get_keypoints(frame_idx, individual, entry)

            from poser.core.frame_export import save_frame_as_yolo
            img_path, label_path = save_frame_as_yolo(
                frame_image=frame_image,
                keypoints=keypoints,
                frame_idx=frame_idx,
                source_stem=entry.stem if entry else "frame",
                output_dir=Path(output_dir),
            )
            QMessageBox.information(self, "Saved", f"Image: {img_path}\nLabel: {label_path}")

        else:  # Classification
            if entry is None or entry.coords_data is None:
                QMessageBox.warning(self, "No data", "No pose data loaded.")
                return

            label, ok = self._ask_label()
            if not ok:
                return

            from poser.core.frame_export import save_frame_as_classification
            h5_path = save_frame_as_classification(
                coords_data=entry.coords_data,
                individual_key=individual,
                frame_idx=frame_idx,
                label=label,
                source_file=entry.pose_path,
                output_dir=Path(output_dir),
            )
            QMessageBox.information(self, "Saved", f"Saved classification frame to:\n{h5_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_bout_count(self) -> None:
        self._bout_count_label.setText(f"Bouts detected: {len(self._detected_bouts)}")

    def _get_frame_image(self, frame_idx: int):
        """Try to extract a 2-D image array from the active image layer."""
        for layer in self._viewer.layers:
            try:
                import napari.layers
                if isinstance(layer, napari.layers.Image):
                    data = layer.data
                    if data.ndim >= 3:
                        return np.array(data[frame_idx])
                    return np.array(data)
            except Exception:
                continue
        return None

    def _get_keypoints(self, frame_idx: int, individual: str, entry) -> Optional[np.ndarray]:
        """Return ``(V, 2)`` keypoints for *frame_idx* from session coords_data."""
        if entry is None or entry.coords_data is None:
            return None
        ind = entry.coords_data.get(individual)
        if ind is None:
            return None
        x = np.array(ind["x"])
        y = np.array(ind["y"])
        if frame_idx >= len(x):
            return None
        return np.stack([x[frame_idx], y[frame_idx]], axis=-1)  # (V, 2)

    def _ask_label(self):
        """Simple input dialog to ask for a behaviour label index."""
        from qtpy.QtWidgets import QInputDialog
        val, ok = QInputDialog.getInt(self, "Label", "Behaviour label index:", 0, 0, 999)
        return val, ok

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_individuals(self, individuals: List[str]) -> None:
        """Update the individual combo box."""
        self._individual_combo.clear()
        for ind in individuals:
            self._individual_combo.addItem(ind)

    @property
    def bouts(self) -> List[dict]:
        return list(self._detected_bouts)
