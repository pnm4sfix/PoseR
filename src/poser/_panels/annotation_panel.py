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
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import napari
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QKeySequence
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
    QShortcut,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from poser.core.scoring import (
    bout_classes,
    class_names_from_checkpoint,
    labels_for_classes,
)
from poser.core.session import SessionEntry, SessionManager
from poser.core.settings import resolve_output_dir

log = logging.getLogger(__name__)


class AnnotationPanel(QWidget):
    """Widget for manually labelling behaviour bouts.

    Signals
    -------
    annotations_changed(np.ndarray | None)
        Emitted with a per-frame label array (dtype int64) whenever a bout
        label is assigned or the bout list is reloaded, and with None when no
        bouts are listed.  Connected by the factory to
        ``EthogramPanel.load_annotations``.
    labels_changed(dict)
        Emitted with a copy of the label map {index: name} whenever it changes,
        and whenever an entry is shown, which Load Session does. Connected by
        the factory to EthogramPanel.set_annotation_names.
    """

    annotations_changed = Signal(object)
    labels_changed = Signal(object)

    def __init__(
        self,
        viewer: napari.Viewer,
        session: Optional[SessionManager] = None,
        behaviour_labels: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self._viewer = viewer
        self._session = session or SessionManager(viewer=viewer)
        self._entry: SessionEntry | None = None
        self._current_bouts: List[dict] = []
        # A decoder's per-frame predictions for the shown entry, and its checkpoint.
        self._predictions: np.ndarray | None = None
        self._checkpoint: Path | None = None

        self._build_ui()
        if behaviour_labels:
            self.set_behaviour_labels(behaviour_labels)

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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._label_buttons_widget)
        scroll.setMaximumHeight(150)
        label_lay.addWidget(scroll)

        self._label_from_predictions_button = QPushButton(
            "Label bouts from predictions"
        )
        self._label_from_predictions_button.setToolTip(
            "Gives each bout without a hand label the class the decoder predicted most "
            "often in it. Predict in the Inference panel first."
        )
        self._label_from_predictions_button.setEnabled(False)
        self._label_from_predictions_button.clicked.connect(
            self._on_label_from_predictions
        )
        label_lay.addWidget(self._label_from_predictions_button)

        root.addWidget(label_grp)

        # Digit keys label bouts only while this panel has focus: in the viewer,
        # napari uses them to switch layer modes.
        for idx in range(10):
            shortcut = QShortcut(QKeySequence(str(idx)), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            # A held key would otherwise label every bout it auto-advances to.
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(partial(self._assign_label, idx))

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

        remove_row = QHBoxLayout()
        self._remove_label_combo = QComboBox()
        self._remove_label_combo.setToolTip(
            "Behaviour label to remove; the bouts it labels become unlabelled."
        )
        remove_row.addWidget(self._remove_label_combo)
        btn_remove_label = QPushButton("Remove")
        btn_remove_label.clicked.connect(self._on_remove_label)
        remove_row.addWidget(btn_remove_label)
        manage_lay.addLayout(remove_row)
        self._rebuild_label_buttons()

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
        """Rebuild the label buttons and the list of labels to remove."""
        # Remove old buttons
        for i in reversed(range(self._label_buttons_layout.count())):
            w = self._label_buttons_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        chosen = self._remove_label_combo.currentData()
        self._remove_label_combo.clear()
        for idx, name in sorted(self._session.behaviour_labels.items()):
            btn = QPushButton(f"[{idx}] {name}")
            btn.setProperty("label_idx", idx)
            btn.clicked.connect(self._on_assign_label)
            self._label_buttons_layout.addWidget(btn)
            if idx >= 1:  # label 0 (other) cannot be removed
                self._remove_label_combo.addItem(f"[{idx}] {name}", idx)
        # This runs whenever bouts are shown, so keep the label chosen for removal.
        self._remove_label_combo.setCurrentIndex(
            max(self._remove_label_combo.findData(chosen), 0)
        )

    # ------------------------------------------------------------------
    # Public API called by other panels
    # ------------------------------------------------------------------

    def show_entry(self, entry: SessionEntry | None) -> None:
        """List the bouts of entry, or none; labels go to the entry's own bouts."""
        log.debug(
            "Showing the %d bouts of %s; showed %s before",
            len(entry.bouts) if entry is not None else 0,
            entry.pose_path if entry is not None else None,
            self._entry.pose_path if self._entry is not None else None,
        )
        if entry is not self._entry:
            # A bout of another recording over the same frames is another bout, so
            # it must not stay selected.
            self._bout_list.clear()
            # Predictions label the frames of one recording, never another's bouts.
            if self._predictions is not None:
                log.debug(
                    "Dropped the predictions from %s: they are not for %s",
                    self._checkpoint,
                    entry.pose_path if entry is not None else None,
                )
            self._predictions = None
            self._checkpoint = None
            self._label_from_predictions_button.setEnabled(False)
        self._entry = entry
        # Load Session replaces the session's label map and then sends its active
        # entry here, so the buttons are rebuilt from the map now in use.
        self._rebuild_label_buttons()
        self.labels_changed.emit(dict(self._session.behaviour_labels))
        self.load_bouts(entry.bouts if entry is not None else [])

    def load_bouts(self, bouts: List[dict]) -> None:
        """Populate the bout list from detected bouts and select one.

        Selects the bout selected before if one with its start and end is listed,
        else the first unlabelled bout, else the first; the viewer does not move.
        """
        selected = self._bout_list.currentItem()
        data = selected.data(Qt.UserRole) if selected is not None else None
        # PySide returns a stored tuple as a list; the spans compared below are tuples.
        previous = tuple(data) if data is not None else None
        log.debug(
            "load_bouts received %d bouts; label map %s; selected before: %s",
            len(bouts),
            self._session.behaviour_labels,
            previous,
        )
        self._current_bouts = bouts
        self._bout_list.clear()
        for i, b in enumerate(bouts):
            label_idx = b.get("label", -1)
            label_name = self._session.behaviour_labels.get(label_idx, "unlabelled")
            if b.get("label_source") == "decoder":
                label_name += ", decoder"
            item = QListWidgetItem(
                f"Bout {i+1:03d}  frames {b['start']}–{b['end']}  [{label_name}]"
            )
            item.setData(Qt.UserRole, (b["start"], b["end"]))
            self._bout_list.addItem(item)
        self._update_stats()
        labelled = sum(1 for b in bouts if b.get("label", -1) >= 0)
        log.info(
            "Loaded %d bouts into the annotation list: %d labelled, %d unlabelled",
            len(bouts),
            labelled,
            len(bouts) - labelled,
        )
        known = {-1, *self._session.behaviour_labels}
        unknown = {b.get("label", -1) for b in bouts} - known
        if unknown:
            log.warning(
                "Bout labels %s are not in the label map %s; those bouts show as "
                "unlabelled",
                unknown,
                self._session.behaviour_labels,
            )
        spans = [(b["start"], b["end"]) for b in bouts]
        if previous in spans:
            row = spans.index(previous)
        else:
            row = next((i for i, b in enumerate(bouts) if b.get("label", -1) < 0), 0)
        if bouts:
            # With signals blocked this selection does not move the viewer, which
            # only a user's selection should do.
            self._bout_list.blockSignals(True)
            self._bout_list.setCurrentRow(row)
            self._bout_list.blockSignals(False)
            self._on_bout_selected(row, seek=False)
        else:
            self._current_label_display.setText("Current: —")
        self._emit_annotations()
    def set_behaviour_labels(self, labels: Dict[int, str]) -> None:
        """Update label schema and rebuild buttons; label 0 stays other."""
        if labels.get(0, "other") != "other":
            log.warning(
                "Label 0 is always other; the name %r given for it is not used",
                labels[0],
            )
        behaviours = {idx: name for idx, name in labels.items() if idx != 0}
        self._session.behaviour_labels = {0: "other", **behaviours}
        log.debug(
            "Behaviour labels set to %s from %s", self._session.behaviour_labels, labels
        )
        self._rebuild_label_buttons()
        self.labels_changed.emit(dict(self._session.behaviour_labels))

    def resend(self) -> None:
        """Send the names, then the per-frame labels, to a panel just connected."""
        log.debug(
            "Sending the label names %s and the labels of %d bouts again",
            self._session.behaviour_labels,
            len(self._current_bouts),
        )
        self.labels_changed.emit(dict(self._session.behaviour_labels))
        self._emit_annotations()

    def set_predictions(
        self, predictions: np.ndarray, checkpoint: Path | None = None
    ) -> None:
        """Keep a decoder's per-frame predictions to label the shown entry's bouts."""
        self._predictions = np.asarray(predictions)
        self._checkpoint = checkpoint
        self._label_from_predictions_button.setEnabled(True)
        log.info(
            "Kept %d frames of predictions from %s to label the bouts of %s",
            len(self._predictions),
            checkpoint,
            self._entry.pose_path if self._entry is not None else None,
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load_from_session(self) -> None:
        """Load bouts saved by the Analysis panel onto the active session entry."""
        entry = self._session.active
        self.show_entry(entry)
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

    def _on_add_label(self) -> None:
        name = self._new_label_edit.text().strip()
        if not name:
            log.warning("No behaviour label added: the label name is empty")
            return
        next_idx = max(self._session.behaviour_labels.keys(), default=-1) + 1
        self._session.behaviour_labels[next_idx] = name
        self._new_label_edit.clear()
        self._rebuild_label_buttons()
        self.labels_changed.emit(dict(self._session.behaviour_labels))
        log.info(
            "Added behaviour label %s (%s); labels are now %s",
            next_idx,
            name,
            self._session.behaviour_labels,
        )

    def _on_remove_label(self) -> None:
        """Remove the label chosen in the list and unlabel the bouts it labels."""
        labels = self._session.behaviour_labels
        idx = self._remove_label_combo.currentData()
        behaviours = [i for i in labels if i >= 1]
        log.debug("Remove label %s chosen; behaviour labels %s", idx, behaviours)
        if len(behaviours) <= 1:
            log.warning(
                "Behaviour label %s (%s) not removed: at least one behaviour label is "
                "required",
                idx,
                labels.get(idx),
            )
            QMessageBox.warning(
                self, "Cannot remove", "At least one behaviour label is required."
            )
            return
        name = labels[idx]
        # The label map is the session's, so bouts of every entry may carry it.
        bout_lists = [self._current_bouts, *(e.bouts for e in self._session.entries)]
        # The listed bouts are usually an entry's own list; keying by id counts once.
        carrying = {
            id(b): b for bouts in bout_lists for b in bouts if b.get("label") == idx
        }
        if carrying:
            n = len(carrying)
            reply = QMessageBox.question(
                self,
                "Remove label",
                f"{n} {'bout is' if n == 1 else 'bouts are'} labelled {name} in this "
                f"session. Remove the label and leave {'it' if n == 1 else 'them'} "
                "unlabelled?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                log.debug("Removing label %s (%s) cancelled at the question", idx, name)
                return
        for bout in carrying.values():
            del bout["label"]
            bout.pop("label_source", None)
        del labels[idx]
        log.info(
            "Removed behaviour label %s (%s) and unlabelled its %d bouts; labels are "
            "now %s",
            idx,
            name,
            len(carrying),
            labels,
        )
        self._rebuild_label_buttons()
        self.load_bouts(self._current_bouts)
        # Sent after the bouts' new labels: the other way round, the Ethogram would
        # draw the removed label under names that no longer have it.
        self.labels_changed.emit(dict(self._session.behaviour_labels))

    def _on_bout_selected(self, row: int, seek: bool = True) -> None:
        if row < 0 or row >= len(self._current_bouts):
            log.debug(
                "No bout selected (row %d of %d bouts)", row, len(self._current_bouts)
            )
            return
        bout = self._current_bouts[row]
        # Seek viewer to start of bout
        if seek:
            try:
                self._viewer.dims.set_point(0, bout["start"])
            except Exception:
                log.debug(
                    "Cannot move the viewer to bout %d; it stays on its current frame",
                    row + 1,
                    exc_info=True,
                )
        label_idx = bout.get("label", -1)
        label_name = self._session.behaviour_labels.get(label_idx, "unlabelled")
        log.debug(
            "Bout %d selected: frames %s-%s, label %s (%s); move viewer: %s",
            row + 1,
            bout.get("start"),
            bout.get("end"),
            label_idx,
            label_name,
            seek,
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
        """Label the selected bout with the clicked button's label."""
        self._assign_label(self.sender().property("label_idx"))

    def _assign_label(self, label_idx: int) -> None:
        """Label the selected bout by hand, for a label button or digit key."""
        row = self._bout_list.currentRow()
        log.debug(
            "Label %s asked for list row %d (%d bouts listed)",
            label_idx,
            row,
            len(self._current_bouts),
        )
        if label_idx not in self._session.behaviour_labels:
            log.warning(
                "Label not assigned: there is no label %s in the label map %s",
                label_idx,
                self._session.behaviour_labels,
            )
            return
        if row < 0 or row >= len(self._current_bouts):
            log.warning(
                "Label not assigned: no bout selected (row %d, %d bouts loaded)",
                row,
                len(self._current_bouts),
            )
            self._current_label_display.setText("Select a bout first")
            return
        label_name = self._session.behaviour_labels[label_idx]

        self._current_bouts[row]["label"] = label_idx
        self._current_bouts[row]["label_source"] = "hand"
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

    def _on_label_from_predictions(self) -> None:
        """Label each bout without a hand label with the class predicted most in it."""
        bouts = self._current_bouts
        log.debug(
            "Label bouts from predictions: %d bouts, %d frames of predictions from %s",
            len(bouts),
            len(self._predictions),
            self._checkpoint,
        )
        if not bouts:
            log.warning("No bouts labelled from predictions: no bouts are listed")
            self._current_label_display.setText("No bouts to label")
            return
        classes = None
        if self._checkpoint is not None:
            try:
                import torch  # lazy: torch is slow to import

                raw = torch.load(
                    str(self._checkpoint), map_location="cpu", weights_only=False
                )
            except Exception as exc:
                log.exception(
                    "No bouts labelled from predictions: cannot read %s",
                    self._checkpoint,
                )
                QMessageBox.warning(
                    self,
                    "Cannot read the decoder",
                    f"No bouts labelled: cannot read {self._checkpoint}.\n{exc}",
                )
                return
            classes = class_names_from_checkpoint(raw)
        votes = bout_classes(self._predictions, bouts)
        to_label = self._map_classes_to_labels(
            classes.names if classes is not None else None, votes
        )
        labelled = hand = 0
        for bout, cls in zip(bouts, votes, strict=True):
            # A label of no known source may be the user's, so it is kept too.
            if bout.get("label", -1) >= 0 and bout.get("label_source") != "decoder":
                hand += 1
            elif cls is not None:
                bout["label"] = to_label[cls]
                bout["label_source"] = "decoder"
                labelled += 1
        log.info(
            "Labelled %d of %d bouts from the predictions of %s (decoder class to "
            "label %s); kept %d hand labels; %d bouts have no predicted frame",
            labelled,
            len(bouts),
            self._checkpoint,
            to_label,
            hand,
            len(bouts) - labelled - hand,
        )
        self.load_bouts(bouts)
        self._current_label_display.setText(
            f"Labelled {labelled} bouts from predictions; kept {hand} hand labels"
        )

    def _map_classes_to_labels(
        self, names: dict[int, str] | None, votes: list[int | None]
    ) -> dict[int, int]:
        """Map decoder classes to label indices, adding labels for new class names."""
        labels = self._session.behaviour_labels
        voted = {c for c in votes if c is not None}
        log.debug(
            "Mapping decoder classes %s (names %s) to the labels %s",
            sorted(voted),
            names,
            labels,
        )
        before = set(labels)
        to_label = labels_for_classes(names, voted, labels)
        added = {i: labels[i] for i in sorted(labels.keys() - before)}
        if added:
            log.info(
                "Added the decoder's classes as labels %s; labels are now %s",
                added,
                labels,
            )
            self._rebuild_label_buttons()
            self.labels_changed.emit(dict(labels))
        return to_label

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
        entry = self._entry
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
        if labels is None:
            # Otherwise the Ethogram keeps the labels of the entry shown before.
            log.debug("Emitting annotations_changed with None: no bouts are listed")
            self.annotations_changed.emit(None)
            return
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

        entry = self._entry
        if entry is None or not entry.coords_data:
            log.warning(
                "Training data not exported: no pose data on the shown entry (%s)",
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
            #  0 = other (not in any bout, or in a bout labelled other)
            # >0 = class label
            labels = np.zeros(T, dtype=np.int64)
            for b in self._current_bouts:
                lbl = b.get("label", -1)
                start = max(0, int(b["start"]))
                end = min(T, int(b["end"]))
                labels[start:end] = lbl  # -1 preserved for unlabelled bouts

            schema = dict(self._session.behaviour_labels)  # label 0 is always "other"

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
        by_decoder = sum(
            1
            for b in self._current_bouts
            if b.get("label", -1) >= 0 and b.get("label_source") == "decoder"
        )
        counts: Dict[int, int] = {}
        for b in self._current_bouts:
            lbl = b.get("label", -1)
            counts[lbl] = counts.get(lbl, 0) + 1

        parts = [f"{self._session.behaviour_labels.get(k, k)}: {v}"
                 for k, v in counts.items() if k >= 0]
        decoder_note = f" ({by_decoder} by decoder)" if by_decoder else ""
        self._stats_label.setText(
            f"{labelled}/{total} labelled{decoder_note}  |  " + ",  ".join(parts)
        )
