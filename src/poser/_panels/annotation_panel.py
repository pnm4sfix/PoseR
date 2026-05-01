"""
_panels/annotation_panel.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Annotation Panel — apply behaviour labels to detected bouts.

Features:
* Shows the list of bouts for the active session entry
* Play / seek to a bout in the viewer
* Assign a label from the behaviour schema by button-click or keyboard shortcut
* Export labelled bouts to HDF5 for training
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import napari
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from poser.core.session import SessionManager


class AnnotationPanel(QWidget):
    """Widget for manually labelling behaviour bouts.

    Signals
    -------
    annotations_changed(np.ndarray)
        Emitted with a per-frame label array (dtype int64) whenever a bout
        label is assigned or the bout list is reloaded.  Connected by the
        factory to ``EthogramPanel.load_annotations``.
    """

    annotations_changed = Signal(object)

    def __init__(
        self,
        viewer: napari.Viewer,
        session: Optional[SessionManager] = None,
        behaviour_labels: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self._viewer = viewer
        self._session = session or SessionManager(viewer=viewer)
        self._behaviour_labels: Dict[int, str] = behaviour_labels or {0: "behaviour_0", 1: "behaviour_1"}
        self._current_bouts: List[dict] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Load bouts from Analysis panel ───────────────────────────
        btn_load = QPushButton("↻  Load Bouts from Analysis Panel")
        btn_load.setToolTip("Reads bouts saved by the Analysis panel for the active session entry.")
        btn_load.clicked.connect(self._on_load_from_session)
        root.addWidget(btn_load)

        # ── Bout list ─────────────────────────────────────────────────
        bout_grp = QGroupBox("Detected Bouts")
        bout_lay = QVBoxLayout(bout_grp)

        self._bout_list = QListWidget()
        self._bout_list.setMinimumHeight(160)
        self._bout_list.currentRowChanged.connect(self._on_bout_selected)
        bout_lay.addWidget(self._bout_list)

        nav_row = QHBoxLayout()
        btn_prev = QPushButton("◀ Prev")
        btn_prev.clicked.connect(self._on_prev_bout)
        btn_next = QPushButton("Next ▶")
        btn_next.clicked.connect(self._on_next_bout)
        nav_row.addWidget(btn_prev)
        nav_row.addWidget(btn_next)
        bout_lay.addLayout(nav_row)

        root.addWidget(bout_grp)

        # ── Label assignment ──────────────────────────────────────────
        label_grp = QGroupBox("Assign Label")
        label_lay = QVBoxLayout(label_grp)

        self._label_buttons_widget = QWidget()
        self._label_buttons_layout = QVBoxLayout(self._label_buttons_widget)
        self._rebuild_label_buttons()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._label_buttons_widget)
        scroll.setMaximumHeight(150)
        label_lay.addWidget(scroll)

        root.addWidget(label_grp)

        # ── Manage Labels ─────────────────────────────────────────────
        manage_grp = QGroupBox("Manage Labels")
        manage_lay = QVBoxLayout(manage_grp)

        manage_info = QLabel("Add or remove behaviour labels.")
        manage_info.setStyleSheet("font-size: 10px; color: grey;")
        manage_lay.addWidget(manage_info)

        add_row = QHBoxLayout()
        self._new_label_edit = QLineEdit()
        self._new_label_edit.setPlaceholderText("Label name (e.g. swim)")
        add_row.addWidget(self._new_label_edit)
        btn_add_label = QPushButton("Add")
        btn_add_label.clicked.connect(self._on_add_label)
        add_row.addWidget(btn_add_label)
        manage_lay.addLayout(add_row)

        btn_remove_label = QPushButton("Remove Selected Label")
        btn_remove_label.clicked.connect(self._on_remove_label)
        manage_lay.addWidget(btn_remove_label)

        root.addWidget(manage_grp)

        # ── Current assignment display ────────────────────────────────
        self._current_label_display = QLabel("Current: —")
        self._current_label_display.setStyleSheet("font-weight: bold;")
        root.addWidget(self._current_label_display)

        # ── Statistics ────────────────────────────────────────────────
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("font-size: 10px; color: grey;")
        root.addWidget(self._stats_label)

        # ── Export ────────────────────────────────────────────────────
        export_grp = QGroupBox("Export Labels")
        export_lay = QHBoxLayout(export_grp)
        btn_export = QPushButton("Save Training Data (.npy)…")
        btn_export.clicked.connect(self._on_export)
        export_lay.addWidget(btn_export)
        root.addWidget(export_grp)

        root.addStretch()

    def _rebuild_label_buttons(self) -> None:
        # Remove old buttons
        for i in reversed(range(self._label_buttons_layout.count())):
            w = self._label_buttons_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        for idx, name in sorted(self._behaviour_labels.items()):
            btn = QPushButton(f"[{idx}] {name}")
            btn.setProperty("label_idx", idx)
            btn.clicked.connect(self._on_assign_label)
            self._label_buttons_layout.addWidget(btn)

    # ------------------------------------------------------------------
    # Public API called by other panels
    # ------------------------------------------------------------------

    def load_bouts(self, bouts: List[dict]) -> None:
        """Populate the bout list from detected bouts."""
        self._current_bouts = bouts
        self._bout_list.clear()
        for i, b in enumerate(bouts):
            label_idx = b.get("label", -1)
            label_name = self._behaviour_labels.get(label_idx, "unlabelled")
            item = QListWidgetItem(
                f"Bout {i+1:03d}  frames {b['start']}–{b['end']}  [{label_name}]"
            )
            self._bout_list.addItem(item)
        self._update_stats()
        self._emit_annotations()
    def set_behaviour_labels(self, labels: Dict[int, str]) -> None:
        """Update label schema and rebuild buttons."""
        self._behaviour_labels = labels
        self._rebuild_label_buttons()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load_from_session(self) -> None:
        """Load bouts saved by the Analysis panel onto the active session entry."""
        entry = self._session.active
        if entry is None or not entry.bouts:
            QMessageBox.information(
                self, "No bouts",
                "No bouts found on the active session entry.\n"
                "Use the Analysis panel to detect or manually define bouts first."
            )
            return
        self.load_bouts(entry.bouts)

    def _on_add_label(self) -> None:
        name = self._new_label_edit.text().strip()
        if not name:
            return
        next_idx = max(self._behaviour_labels.keys(), default=-1) + 1
        self._behaviour_labels[next_idx] = name
        self._new_label_edit.clear()
        self._rebuild_label_buttons()

    def _on_remove_label(self) -> None:
        """Remove the highest-index label (simple remove-last approach)."""
        if not self._behaviour_labels:
            return
        if len(self._behaviour_labels) == 1:
            QMessageBox.warning(self, "Cannot remove", "At least one label is required.")
            return
        last_idx = max(self._behaviour_labels.keys())
        del self._behaviour_labels[last_idx]
        self._rebuild_label_buttons()

    def _on_bout_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._current_bouts):
            return
        bout = self._current_bouts[row]
        # Seek viewer to start of bout
        try:
            self._viewer.dims.set_point(0, bout["start"])
        except Exception:
            pass
        label_idx = bout.get("label", -1)
        label_name = self._behaviour_labels.get(label_idx, "unlabelled")
        self._current_label_display.setText(f"Current: [{label_idx}] {label_name}")

    def _on_prev_bout(self) -> None:
        row = self._bout_list.currentRow()
        if row > 0:
            self._bout_list.setCurrentRow(row - 1)

    def _on_next_bout(self) -> None:
        row = self._bout_list.currentRow()
        if row < self._bout_list.count() - 1:
            self._bout_list.setCurrentRow(row + 1)

    def _on_assign_label(self) -> None:
        row = self._bout_list.currentRow()
        if row < 0 or row >= len(self._current_bouts):
            return
        btn = self.sender()
        label_idx = btn.property("label_idx")
        label_name = self._behaviour_labels.get(label_idx, str(label_idx))

        self._current_bouts[row]["label"] = label_idx
        item = self._bout_list.item(row)
        b = self._current_bouts[row]
        item.setText(f"Bout {row+1:03d}  frames {b['start']}–{b['end']}  [{label_name}]")

        self._current_label_display.setText(f"Assigned: [{label_idx}] {label_name}")
        self._update_stats()
        self._emit_annotations()

        # Auto-advance to next bout
        if row + 1 < self._bout_list.count():
            self._bout_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # Ethogram integration
    # ------------------------------------------------------------------

    def _bouts_to_labels(self) -> Optional[np.ndarray]:
        """Convert current bouts to a dense per-frame label array.

        Returns ``None`` if there are no bouts.  Frames not covered by any
        bout are assigned label 0 (background / other).
        """
        if not self._current_bouts:
            return None
        # Try to get total frame count from the session entry first
        T: Optional[int] = None
        entry = self._session.active
        if entry is not None and entry.coords_data:
            try:
                individual = next(iter(entry.coords_data))
                x_data = entry.coords_data[individual]["x"]
                T = len(x_data[0]) if hasattr(x_data[0], "__len__") else np.array(x_data).shape[-1]
            except Exception:
                pass
        if T is None:
            T = max(int(b["end"]) for b in self._current_bouts) + 1
        labels = np.zeros(T, dtype=np.int64)
        for b in self._current_bouts:
            lbl = b.get("label", -1)
            # -1 means detected but not yet labelled — drawn as gray in ethogram
            labels[max(0, int(b["start"])) : min(T, int(b["end"]))] = lbl
        return labels

    def _emit_annotations(self) -> None:
        """Build per-frame label array and emit ``annotations_changed``."""
        labels = self._bouts_to_labels()
        if labels is not None:
            self.annotations_changed.emit(labels)

    def _on_export(self) -> None:
        if not self._current_bouts:
            QMessageBox.warning(self, "No bouts", "No bouts to export.")
            return

        unlabelled = [b for b in self._current_bouts if b.get("label", -1) < 0]
        if unlabelled:
            reply = QMessageBox.question(
                self, "Unlabelled bouts",
                f"{len(unlabelled)} bouts are still unlabelled. "
                "Unlabelled frames will default to label 0 (other). Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        folder = QFileDialog.getExistingDirectory(
            self, "Select output folder for training data"
        )
        if not folder:
            return

        entry = self._session.active
        if entry is None or not entry.coords_data:
            QMessageBox.warning(self, "No pose data", "No pose data on the active entry.")
            return

        try:
            import json
            import numpy as np
            from pathlib import Path

            out = Path(folder)
            individual = list(entry.coords_data.keys())[0]
            ind_data = entry.coords_data[individual]

            # Build CTVM array: C=3 (x, y, ci), T=frames, V=nodes, M=1
            x = np.array(ind_data["x"])   # (V, T)
            y = np.array(ind_data["y"])   # (V, T)
            ci = np.array(ind_data["ci"]) # (V, T)
            V, T = x.shape

            # Stack to (C, T, V, M=1)
            pose = np.stack([x, y, ci], axis=0)  # (3, V, T)
            pose = pose.transpose(0, 2, 1)        # (3, T, V)
            pose = pose[:, :, :, np.newaxis]       # (3, T, V, 1) = (C, T, V, M)

            # Build per-frame label array
            # -1 = detected but unlabelled (ignored during training)
            #  0 = background (not in any bout)
            # >0 = class label
            labels = np.zeros(T, dtype=np.int64)
            for b in self._current_bouts:
                lbl = b.get("label", -1)
                start = max(0, int(b["start"]))
                end = min(T, int(b["end"]))
                labels[start:end] = lbl  # -1 preserved for unlabelled bouts

            # Build schema — ensure 0 = "other"
            schema = {0: "other"}
            for idx, name in self._behaviour_labels.items():
                if idx != 0:
                    schema[idx] = name
                elif name not in ("behaviour_0", "other"):
                    schema[0] = name

            stem = Path(entry.pose_path).stem if entry.pose_path else "recording"
            np.save(str(out / f"{stem}_pose.npy"), pose)
            np.save(str(out / f"{stem}_labels.npy"), labels)
            with open(out / f"{stem}_schema.json", "w") as fh:
                json.dump({str(k): v for k, v in schema.items()}, fh, indent=2)

        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        labelled_frames = int((labels > 0).sum())
        unlabelled_frames = int((labels == -1).sum())
        QMessageBox.information(
            self, "Saved",
            f"Saved to: {folder}\n\n"
            f"  {stem}_pose.npy    \u2014 shape {pose.shape}  (C, T, V, M)\n"
            f"  {stem}_labels.npy  \u2014 {labelled_frames}/{T} frames labelled"
            + (f", {unlabelled_frames} unlabelled (-1)" if unlabelled_frames else "") + "\n"
            f"  {stem}_schema.json \u2014 {len(schema)} classes\n\n"
            f"Unlabelled frames (-1) are ignored by the trainer.\n"
            f"Load for training with:\n"
            f"  PoseDataset(data_file='{stem}_pose.npy',\n"
            f"              label_file='{stem}_labels.npy',\n"
            f"              preprocess_frame=True, window_size=12)"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_stats(self) -> None:
        total = len(self._current_bouts)
        labelled = sum(1 for b in self._current_bouts if b.get("label", -1) >= 0)
        counts: Dict[int, int] = {}
        for b in self._current_bouts:
            lbl = b.get("label", -1)
            counts[lbl] = counts.get(lbl, 0) + 1

        parts = [f"{self._behaviour_labels.get(k, k)}: {v}"
                 for k, v in counts.items() if k >= 0]
        self._stats_label.setText(
            f"{labelled}/{total} labelled  |  " + ",  ".join(parts)
        )
