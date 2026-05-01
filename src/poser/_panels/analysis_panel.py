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
from typing import Callable, Dict, List, Optional

import napari
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
        viewer: napari.Viewer,
        session: Optional[SessionManager] = None,
        on_bouts_updated: Optional[Callable[[List[dict]], None]] = None,
    ):
        super().__init__()
        self._viewer = viewer
        self._session = session or SessionManager(viewer=viewer)
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

        # Center node
        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("Center node index:"))
        self._center_node_spin = QSpinBox()
        self._center_node_spin.setRange(0, 999)
        self._center_node_spin.setValue(0)
        self._center_node_spin.setToolTip(
            "Index of the body-part used as the reference/centre node "
            "for egocentric and orthogonal projection calculations."
        )
        center_row.addWidget(self._center_node_spin)
        params_lay.addLayout(center_row)

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
        entry = self._session.active
        coords_data = entry.coords_data if entry else None
        if not coords_data:
            QMessageBox.warning(self, "No data", "Load a pose file first.")
            return

        # Sync combo with actual keys in coords_data (DLC uses 'individual1' etc.)
        available = list(coords_data.keys())
        individual = self._individual_combo.currentText()
        if set(available) != {self._individual_combo.itemText(i)
                               for i in range(self._individual_combo.count())}:
            self._individual_combo.blockSignals(True)
            self._individual_combo.clear()
            self._individual_combo.addItems(available)
            if individual in available:
                self._individual_combo.setCurrentText(individual)
            self._individual_combo.blockSignals(False)
        individual = self._individual_combo.currentText()
        if individual not in coords_data:
            individual = available[0]
            self._individual_combo.setCurrentText(individual)

        bout = manual_bout(
            start=self._manual_start,
            end=end_frame,
            coords_data=coords_data,
            individual_key=individual,
        )
        self._detected_bouts.append(bout)
        self._update_bout_count()
        self._save_bouts_to_session(self._detected_bouts)
        if self._on_bouts_updated:
            self._on_bouts_updated(list(self._detected_bouts))

        self._manual_range_label.setText(
            f"Added: {self._manual_start} → {end_frame}  "
            f"(total {len(self._detected_bouts)} bouts)"
        )
        self._manual_start = None
        self._btn_set_end.setEnabled(False)

    def _on_detect(self) -> None:
        entry = self._session.active
        if entry is None or not entry.coords_data:
            QMessageBox.warning(self, "No data", "Load a pose file first.")
            return

        from poser.core.bout_detection import orthogonal_variance, egocentric_variance

        coords_data = entry.coords_data

        # Keep the combo in sync with whatever individuals are in coords_data
        available = list(coords_data.keys())
        current = self._individual_combo.currentText()
        if set(available) != {self._individual_combo.itemText(i)
                               for i in range(self._individual_combo.count())}:
            self._individual_combo.blockSignals(True)
            self._individual_combo.clear()
            self._individual_combo.addItems(available)
            if current in available:
                self._individual_combo.setCurrentText(current)
            self._individual_combo.blockSignals(False)

        individual = self._individual_combo.currentText()
        if individual not in coords_data:
            individual = available[0]
            self._individual_combo.setCurrentText(individual)

        ind = coords_data[individual]
        x = np.array(ind["x"]).astype(float)   # (n_nodes, n_frames)
        y = np.array(ind["y"]).astype(float)
        n_nodes, n_frames = x.shape

        # Build flat (n_nodes * n_frames, 3) array with columns (frame, y, x)
        # matching what the bout detection functions expect after reshape(n_nodes, -1, 3)
        frame_idx = np.tile(np.arange(n_frames, dtype=float), n_nodes)
        points = np.column_stack([frame_idx, y.flatten(), x.flatten()])
        np.nan_to_num(points, copy=False)

        fps = self._fps_spin.value()
        method = self._method_combo.currentText()
        center_node = min(self._center_node_spin.value(), n_nodes - 1)

        try:
            if method == "egocentric":
                bouts, *_ = egocentric_variance(
                    points, center_node=center_node, fps=fps, n_nodes=n_nodes
                )
            else:
                bouts, *_ = orthogonal_variance(
                    points, center_node=center_node, fps=fps, n_nodes=n_nodes
                )
        except Exception as exc:
            QMessageBox.critical(self, "Detection error", str(exc))
            return

        # Detection functions return (start, end) tuples — convert to dicts
        bouts = [
            b if isinstance(b, dict) else {"start": int(b[0]), "end": int(b[1])}
            for b in bouts
        ]

        # Filter by duration
        min_f = self._min_frames_spin.value()
        max_f = self._max_frames_spin.value()
        bouts = [b for b in bouts if min_f <= (b["end"] - b["start"]) <= max_f]

        self._detected_bouts = bouts
        self._update_bout_count()
        self._save_bouts_to_session(bouts)
        if self._on_bouts_updated:
            self._on_bouts_updated(list(bouts))

    def _save_bouts_to_session(self, bouts: list) -> None:
        """Persist bouts onto the active SessionEntry so other panels can read them."""
        entry = self._session.active
        if entry is not None:
            entry.bouts = list(bouts)

    def _on_export_frame(self) -> None:
        try:
            frame_idx = int(self._viewer.dims.current_step[0])
        except Exception:
            frame_idx = 0

        entry = self._session.active
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

            # Prefer reading keypoints directly from the visible Points layer
            # (already filtered, NaN-free) then fall back to coords_data
            keypoints = self._get_keypoints_from_layer(frame_idx, individual)
            if keypoints is None:
                keypoints = self._get_keypoints(frame_idx, individual, entry)
            if keypoints is None:
                QMessageBox.warning(
                    self, "No keypoints",
                    f"No pose data found for individual '{individual}' "
                    f"at frame {frame_idx}.\n\n"
                    "Make sure a pose file is loaded and the correct "
                    "individual is selected."
                )
                return

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
        """Try to extract a 2-D image array from the *visible* active image layer."""
        import napari.layers
        # Prefer the visible layer; if none visible, take the first image layer
        candidates = [
            lyr for lyr in self._viewer.layers
            if isinstance(lyr, napari.layers.Image)
        ]
        visible = [lyr for lyr in candidates if lyr.visible]
        layer = visible[0] if visible else (candidates[0] if candidates else None)
        if layer is None:
            return None
        try:
            data = layer.data
            if hasattr(data, 'compute'):   # dask array
                frame = data[frame_idx].compute()
            else:
                frame = np.array(data[frame_idx])
            return frame
        except Exception:
            return None

    def _get_keypoints_from_layer(self, frame_idx: int, individual: str) -> Optional[np.ndarray]:
        """Read (V, 3) keypoints for *frame_idx* from the visible napari Points layer.

        Returns array of shape (V, 3) with columns x, y, confidence, or None.
        """
        import napari.layers
        visible_pts = [
            lyr for lyr in self._viewer.layers
            if isinstance(lyr, napari.layers.Points) and lyr.visible
        ]
        if not visible_pts:
            return None
        layer = visible_pts[0]
        pts = layer.data          # (N, 3): frame, y, x
        props = layer.properties

        frame_mask = pts[:, 0].astype(int) == frame_idx
        if not frame_mask.any():
            return None

        xs = pts[frame_mask, 2]
        ys = pts[frame_mask, 1]
        confs = props.get("confidence", np.ones(len(xs)))[frame_mask]
        inds  = props.get("ind",        np.zeros(len(xs), dtype=int))[frame_mask]

        # map individual name to index
        ind_names = sorted(set(inds))
        try:
            ind_idx = int(individual.replace("ind", "")) - 1
        except Exception:
            ind_idx = 0

        ind_mask = inds == ind_idx
        if not ind_mask.any():
            # fallback: just use all points at this frame
            ind_mask = np.ones(len(xs), dtype=bool)

        return np.stack([xs[ind_mask], ys[ind_mask], confs[ind_mask]], axis=-1)

    def _get_keypoints(self, frame_idx: int, individual: str, entry) -> Optional[np.ndarray]:
        """Return ``(V, 2)`` keypoints for *frame_idx* from session coords_data."""
        if entry is None or not entry.coords_data:
            return None
        ind = entry.coords_data.get(individual)
        if ind is None:
            return None
        import pandas as pd
        x = ind["x"]
        y = ind["y"]
        ci = ind.get("ci")
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x)
            y = pd.DataFrame(y)
            if ci is not None:
                ci = pd.DataFrame(ci)
        n_nodes, n_frames = x.shape
        if frame_idx >= n_frames:
            return None
        xs = x.iloc[:, frame_idx].to_numpy(dtype=float)  # (V,)
        ys = y.iloc[:, frame_idx].to_numpy(dtype=float)  # (V,)
        if ci is not None:
            cs = ci.iloc[:, frame_idx].to_numpy(dtype=float)
            return np.stack([xs, ys, cs], axis=-1)  # (V, 3)
        return np.stack([xs, ys], axis=-1)  # (V, 2)

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
