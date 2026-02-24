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
from typing import TYPE_CHECKING, Dict, List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from poser.core.io import save_to_h5
from poser.core.session import SessionManager

if TYPE_CHECKING:
    import napari


class AnnotationPanel(QWidget):
    """Widget for manually labelling behaviour bouts.

    Parameters
    ----------
    viewer:
        The running ``napari.Viewer`` instance.
    session:
        Shared :class:`~poser.core.session.SessionManager`.
    behaviour_labels:
        Mapping ``{index: label_name}``.
    """

    def __init__(
        self,
        viewer: "napari.Viewer",
        session: SessionManager,
        behaviour_labels: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self._viewer = viewer
        self._session = session
        self._behaviour_labels: Dict[int, str] = behaviour_labels or {0: "behaviour_0", 1: "behaviour_1"}
        self._current_bouts: List[dict] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

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
        btn_export = QPushButton("Save Labelled Bouts (HDF5)…")
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

    def set_behaviour_labels(self, labels: Dict[int, str]) -> None:
        """Update label schema and rebuild buttons."""
        self._behaviour_labels = labels
        self._rebuild_label_buttons()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

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

        # Auto-advance to next bout
        if row + 1 < self._bout_list.count():
            self._bout_list.setCurrentRow(row + 1)

    def _on_export(self) -> None:
        if not self._current_bouts:
            QMessageBox.warning(self, "No bouts", "No bouts to export.")
            return

        unlabelled = [b for b in self._current_bouts if b.get("label", -1) < 0]
        if unlabelled:
            reply = QMessageBox.question(
                self, "Unlabelled bouts",
                f"{len(unlabelled)} bouts are still unlabelled. Export anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save labelled bouts", "classification.h5", "HDF5 (*.h5)"
        )
        if not path:
            return

        entry = self._session.active_entry
        video_file = entry.video_path.name if entry and entry.video_path else "unknown"

        # Build classification_data list expected by save_to_h5
        classification_data = []
        for b in self._current_bouts:
            if b.get("label", -1) >= 0:
                classification_data.append({
                    "coords": b.get("coords"),
                    "label": b["label"],
                    "start": b["start"],
                    "end": b["end"],
                    "individual": b.get("individual", "ind1"),
                })

        n_nodes = self._current_bouts[0].get("coords", [[]])[0].__len__() if self._current_bouts else 9
        save_to_h5(
            classification_data=classification_data,
            video_file=video_file,
            n_nodes=n_nodes,
            behaviour_schema=self._behaviour_labels,
        )
        QMessageBox.information(self, "Saved", f"Exported {len(classification_data)} bouts to:\n{path}")

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
