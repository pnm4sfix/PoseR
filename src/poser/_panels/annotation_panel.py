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

import logging
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
from poser.core.settings import resolve_output_dir

log = logging.getLogger(__name__)


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
        log.debug(
            "load_bouts received %d bouts; label map %s",
            len(bouts),
            self._behaviour_labels,
        )
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
        labelled = sum(1 for b in bouts if b.get("label", -1) >= 0)
        log.info(
            "Loaded %d bouts into the annotation list: %d labelled, %d unlabelled",
            len(bouts),
            labelled,
            len(bouts) - labelled,
        )
        unknown = {b.get("label", -1) for b in bouts} - {-1, *self._behaviour_labels}
        if unknown:
            log.warning(
                "Bout labels %s are not in the label map %s; those bouts show as "
                "unlabelled",
                unknown,
                self._behaviour_labels,
            )
        self._emit_annotations()
    def set_behaviour_labels(self, labels: Dict[int, str]) -> None:
        """Update label schema and rebuild buttons."""
        log.debug("Behaviour labels set to %s", labels)
        self._behaviour_labels = labels
        self._rebuild_label_buttons()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load_from_session(self) -> None:
        """Load bouts saved by the Analysis panel onto the active session entry."""
        entry = self._session.active
        if entry is None or not entry.bouts:
            log.warning(
                "No bouts to load: none on the active session entry (%s); detect or "
                "define bouts in the Analysis panel first",
                entry.pose_path if entry else None,
            )
            QMessageBox.information(
                self, "No bouts",
                "No bouts found on the active session entry.\n"
                "Use the Analysis panel to detect or manually define bouts first."
            )
            return
        log.debug(
            "Loading %d bouts from session entry %s", len(entry.bouts), entry.pose_path
        )
        self.load_bouts(entry.bouts)

    def _on_add_label(self) -> None:
        name = self._new_label_edit.text().strip()
        if not name:
            log.warning("No behaviour label added: the label name is empty")
            return
        next_idx = max(self._behaviour_labels.keys(), default=-1) + 1
        self._behaviour_labels[next_idx] = name
        self._new_label_edit.clear()
        self._rebuild_label_buttons()
        log.info(
            "Added behaviour label %s (%s); labels are now %s",
            next_idx,
            name,
            self._behaviour_labels,
        )

    def _on_remove_label(self) -> None:
        """Remove the highest-index label (simple remove-last approach)."""
        if not self._behaviour_labels:
            log.warning("No behaviour label removed: there are no labels")
            return
        if len(self._behaviour_labels) == 1:
            log.warning("Behaviour label not removed: at least one label is required")
            QMessageBox.warning(self, "Cannot remove", "At least one label is required.")
            return
        last_idx = max(self._behaviour_labels.keys())
        del self._behaviour_labels[last_idx]
        self._rebuild_label_buttons()
        log.info(
            "Removed behaviour label %s; labels are now %s",
            last_idx,
            self._behaviour_labels,
        )

    def _on_bout_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._current_bouts):
            log.debug(
                "No bout selected (row %d of %d bouts)", row, len(self._current_bouts)
            )
            return
        bout = self._current_bouts[row]
        # Seek viewer to start of bout
        try:
            self._viewer.dims.set_point(0, bout["start"])
        except Exception:
            log.debug(
                "Cannot move the viewer to bout %d; it stays on its current frame",
                row + 1,
                exc_info=True,
            )
        label_idx = bout.get("label", -1)
        label_name = self._behaviour_labels.get(label_idx, "unlabelled")
        log.debug(
            "Bout %d selected: frames %s-%s, label %s (%s)",
            row + 1,
            bout.get("start"),
            bout.get("end"),
            label_idx,
            label_name,
        )
        self._current_label_display.setText(f"Current: [{label_idx}] {label_name}")

    def _on_prev_bout(self) -> None:
        row = self._bout_list.currentRow()
        log.debug("Previous bout requested from row %d", row)
        if row > 0:
            self._bout_list.setCurrentRow(row - 1)

    def _on_next_bout(self) -> None:
        row = self._bout_list.currentRow()
        log.debug(
            "Next bout requested from row %d of %d", row, self._bout_list.count()
        )
        if row < self._bout_list.count() - 1:
            self._bout_list.setCurrentRow(row + 1)

    def _on_assign_label(self) -> None:
        row = self._bout_list.currentRow()
        if row < 0 or row >= len(self._current_bouts):
            log.warning(
                "Label not assigned: no bout selected (row %d, %d bouts loaded)",
                row,
                len(self._current_bouts),
            )
            return
        btn = self.sender()
        label_idx = btn.property("label_idx")
        label_name = self._behaviour_labels.get(label_idx, str(label_idx))
        if label_idx not in self._behaviour_labels:
            log.warning(
                "Label index %s is not in the label map %s",
                label_idx,
                self._behaviour_labels,
            )

        self._current_bouts[row]["label"] = label_idx
        item = self._bout_list.item(row)
        b = self._current_bouts[row]
        item.setText(f"Bout {row+1:03d}  frames {b['start']}–{b['end']}  [{label_name}]")

        self._current_label_display.setText(f"Assigned: [{label_idx}] {label_name}")
        log.info(
            "Assigned label %s (%s) to bout %d, frames %s-%s",
            label_idx,
            label_name,
            row + 1,
            b["start"],
            b["end"],
        )
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
            log.debug("No bouts loaded; no per-frame label array")
            return None
        # Try to get total frame count from the session entry first
        T: Optional[int] = None
        entry = self._session.active
        if entry is not None and entry.coords_data:
            try:
                individual = next(iter(entry.coords_data))
                x_data = entry.coords_data[individual]["x"]
                T = np.asarray(x_data).shape[-1]  # x is (V, T), a DataFrame or array
                log.debug(
                    "Label array length T=%d from %s x data on %s (%s, shape %s)",
                    T,
                    individual,
                    entry.pose_path,
                    type(x_data).__name__,
                    getattr(x_data, "shape", None),
                )
            except Exception:
                log.debug(
                    "Cannot read the frame count from the pose data of %s; "
                    "using the last bout end",
                    entry.pose_path,
                    exc_info=True,
                )
        if T is None:
            T = max(int(b["end"]) for b in self._current_bouts) + 1
            log.debug("Label array length T=%d from the last bout end", T)
        labels = np.zeros(T, dtype=np.int64)
        for b in self._current_bouts:
            lbl = b.get("label", -1)
            # -1 means detected but not yet labelled — drawn as gray in ethogram
            labels[max(0, int(b["start"])) : min(T, int(b["end"]))] = lbl
        last_end = max(int(b["end"]) for b in self._current_bouts)
        if last_end > T:
            log.warning(
                "Label array has %d frames, shorter than the last bout end %d; "
                "bout frames from %d on are dropped",
                T,
                last_end,
                T,
            )
        return labels

    def _emit_annotations(self) -> None:
        """Build per-frame label array and emit ``annotations_changed``."""
        labels = self._bouts_to_labels()
        if labels is not None:
            values, counts = np.unique(labels, return_counts=True)
            log.debug(
                "Emitting annotations_changed: %d frames, frames per label %s",
                len(labels),
                dict(zip(values.tolist(), counts.tolist())),
            )
            self.annotations_changed.emit(labels)

    def _on_export(self) -> None:
        if not self._current_bouts:
            log.warning("Training data not exported: no bouts loaded")
            QMessageBox.warning(self, "No bouts", "No bouts to export.")
            return

        unlabelled = [b for b in self._current_bouts if b.get("label", -1) < 0]
        if unlabelled:
            log.debug(
                "%d of %d bouts are unlabelled; asking whether to export anyway",
                len(unlabelled),
                len(self._current_bouts),
            )
            n = len(unlabelled)
            reply = QMessageBox.question(
                self, "Unlabelled bouts",
                f"{n} {'bout is' if n == 1 else 'bouts are'} still unlabelled. "
                f"{'Its' if n == 1 else 'Their'} frames are saved as -1 and ignored in "
                "training. Continue?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                log.debug(
                    "Training data export cancelled at the unlabelled bouts question"
                )
                return

        entry = self._session.active
        if entry is None or not entry.coords_data:
            log.warning(
                "Training data not exported: no pose data on the active entry (%s)",
                entry.pose_path if entry else None,
            )
            QMessageBox.warning(self, "No pose data", "No pose data on the active entry.")
            return

        stem = Path(entry.pose_path).stem if entry.pose_path else "recording"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save training data as",
            str(resolve_output_dir("training") / f"{stem}_pose.npy"),
            "Training pose (*_pose.npy)",
        )
        if not path:
            log.debug("Training data export cancelled in the save dialog")
            return
        # The Train panel pairs <stem>_pose.npy with <stem>_labels.npy, so the
        # chosen name only gives the stem and the suffixes are put back.
        out = Path(path).parent
        stem = Path(path).name.removesuffix(".npy").removesuffix("_pose")
        log.debug("Training data file chosen: %s; saving as %s_*", path, out / stem)

        try:
            import json
            import numpy as np

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

            np.save(str(out / f"{stem}_pose.npy"), pose)
            np.save(str(out / f"{stem}_labels.npy"), labels)
            with open(out / f"{stem}_schema.json", "w") as fh:
                json.dump({str(k): v for k, v in schema.items()}, fh, indent=2)

        except Exception as exc:
            log.exception("Training data export to %s failed", out / stem)
            QMessageBox.critical(self, "Export failed", str(exc))
            return

        labelled_frames = int((labels > 0).sum())
        unlabelled_frames = int((labels == -1).sum())
        last_end = max(int(b["end"]) for b in self._current_bouts)
        if last_end > T:
            log.warning(
                "Training labels have %d frames, shorter than the last bout end %d; "
                "bout frames from %d on are not saved",
                T,
                last_end,
                T,
            )
        log.info(
            "Saved training data from %s (individual %s) to %s_pose.npy, "
            "%s_labels.npy and %s_schema.json: pose %s (C, T, V, M), %d/%d frames "
            "labelled, %d unlabelled (-1), schema %s",
            entry.pose_path,
            individual,
            out / stem,
            out / stem,
            out / stem,
            pose.shape,
            labelled_frames,
            T,
            unlabelled_frames,
            schema,
        )
        values, counts = np.unique(labels, return_counts=True)
        log.debug(
            "Saved frames per label: %s", dict(zip(values.tolist(), counts.tolist()))
        )
        QMessageBox.information(
            self, "Saved",
            f"Saved to: {out}\n\n"
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
