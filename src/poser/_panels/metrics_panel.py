"""Metrics panel: score a decoder's predictions against hand-labelled bouts."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import napari
import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from poser.core.io import read_classification_h5
from poser.core.schemas.scoring import ClassNames, DecoderScore, LabelledBout
from poser.core.scoring import (
    class_names_from_checkpoint,
    class_names_from_config,
    labelled_bouts,
    save_score_csv,
    score_bouts,
)
from poser.core.session import SessionManager
from poser.core.settings import resolve_output_dir

log = logging.getLogger(__name__)

_COUNTS, _PERCENT = "Counts", "% of true class"
_TABLE_COLUMNS = ("Class", "Precision", "Recall", "F1", "Support")


def _input_row(title: str, value: QLabel, button: QPushButton) -> QHBoxLayout:
    """One input: what it is, what is loaded, and its Browse button."""
    row = QHBoxLayout()
    row.addWidget(QLabel(title))
    value.setWordWrap(True)
    value.setStyleSheet("font-size: 10px; color: grey;")
    row.addWidget(value, 1)
    button.setFixedWidth(80)
    row.addWidget(button)
    return row


class MetricsPanel(QWidget):
    """Napari dock widget: a decoder's confusion matrix and scores."""

    def __init__(self, viewer: napari.Viewer, session: SessionManager) -> None:
        super().__init__()
        self._viewer = viewer
        self._session = session
        self._predictions: np.ndarray | None = None
        self._predictions_path: Path | None = None  # None when sent by Inference
        self._bouts: list[LabelledBout] | None = None
        self._classes: ClassNames | None = None
        self._classes_browsed = False  # a browsed choice beats an automatic one
        self._score: DecoderScore | None = None
        self._build_ui()
        self._connect()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_inputs_group())

        self._score_button = QPushButton("Score")
        layout.addWidget(self._score_button)
        self._summary = QLabel("Load predictions, hand labels and class names.")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("font-weight: bold;")
        layout.addWidget(self._summary)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Show:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems([_COUNTS, _PERCENT])
        view_row.addWidget(self._view_combo)
        view_row.addStretch()
        layout.addLayout(view_row)

        self._figure = Figure(facecolor="#1e1e1e")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setMinimumHeight(260)
        layout.addWidget(self._canvas, 1)

        self._table = QTableWidget(0, len(_TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels(_TABLE_COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setFixedHeight(130)
        layout.addWidget(self._table)

        self._export_button = QPushButton("Export PNG + CSV…")
        self._export_button.setEnabled(False)
        layout.addWidget(self._export_button)

        self._status = QTextEdit()
        self._status.setReadOnly(True)
        self._status.setFixedHeight(80)
        # The platform's own fixed-width font. The generic name "monospace" makes
        # Qt scan every installed font on macOS the first time it is shown.
        self._status.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._status.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; }"
        )
        layout.addWidget(self._status)

    def _build_inputs_group(self) -> QGroupBox:
        group = QGroupBox("Inputs")
        layout = QVBoxLayout(group)
        self._predictions_label = QLabel(
            "None yet. The *_predictions.npy file that Predict writes, not the "
            ".ckpt. Easiest: keep this panel open and run Predict in the "
            "Inference panel."
        )
        self._predictions_button = QPushButton("Browse…")
        layout.addLayout(
            _input_row(
                "Predictions:", self._predictions_label, self._predictions_button
            )
        )
        self._labels_label = QLabel(
            "None yet. The *_classification.h5 file with the hand-labelled bouts, "
            "not the pose .h5."
        )
        self._labels_button = QPushButton("Browse…")
        layout.addLayout(
            _input_row("Hand labels:", self._labels_label, self._labels_button)
        )
        self._names_label = QLabel(
            "None yet. The decoder_config.yml the decoder was trained with, or a "
            ".ckpt that stores its class names."
        )
        self._names_button = QPushButton("Browse…")
        layout.addLayout(
            _input_row("Class names:", self._names_label, self._names_button)
        )
        return group

    def _connect(self) -> None:
        self._predictions_button.clicked.connect(self._on_predictions_clicked)
        self._labels_button.clicked.connect(self._on_labels_clicked)
        self._names_button.clicked.connect(self._on_names_clicked)
        self._score_button.clicked.connect(self._on_score_clicked)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        self._export_button.clicked.connect(self._on_export_clicked)

    def load_predictions(
        self, predictions: np.ndarray, checkpoint: Path | None = None
    ) -> None:
        """Take predictions from the Inference panel, and its class names if any."""
        self._predictions = np.asarray(predictions, dtype=np.int64)
        log.debug(
            "load_predictions: shape %s, checkpoint %s, class names browsed %s",
            self._predictions.shape,
            checkpoint,
            self._classes_browsed,
        )
        self._predictions_path = None
        source = f" ({Path(checkpoint).name})" if checkpoint is not None else ""
        self._predictions_label.setText(
            f"From the Inference panel{source}: {len(self._predictions):,} frames."
        )
        labels_present, frame_counts = np.unique(self._predictions, return_counts=True)
        log.info(
            "Metrics received %d frames of predictions (checkpoint %s); "
            "frames per class %s",
            len(self._predictions),
            checkpoint,
            dict(zip(labels_present.tolist(), frame_counts.tolist())),
        )
        if checkpoint is not None and not self._classes_browsed:
            classes = self._classes_from_checkpoint(Path(checkpoint))
            log.debug("Class names stored in %s: %s", checkpoint, classes)
            if classes is not None:
                self._set_classes(classes, f"From {Path(checkpoint).name}")
        self._score_if_ready()

    def _classes_from_checkpoint(self, checkpoint: Path) -> ClassNames | None:
        """The class names a checkpoint stores, or None if it stores none."""
        log.debug("Reading class names from checkpoint %s", checkpoint)
        try:
            raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        except Exception as exc:
            # The scores do not need the checkpoint, only its class names.
            log.exception("Could not read class names from %s", checkpoint)
            self._status.append(
                f"Could not read class names from {checkpoint.name}: {exc}"
            )
            return None
        return class_names_from_checkpoint(raw)

    def _on_predictions_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select predictions", "", "Predictions (*_predictions.npy *.npy)"
        )
        log.debug("Predictions file dialog: %s", path or "cancelled")
        if not path:
            return
        self._predictions = np.load(path).astype(np.int64)
        self._predictions_path = Path(path)
        self._predictions_label.setText(
            f"{Path(path).name}: {len(self._predictions):,} frames."
        )
        labels_present, frame_counts = np.unique(self._predictions, return_counts=True)
        log.info(
            "Loaded %d frames of predictions from %s; frames per class %s",
            len(self._predictions),
            path,
            dict(zip(labels_present.tolist(), frame_counts.tolist())),
        )
        self._score_if_ready()

    def _on_labels_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select hand labels",
            "",
            "Classification files (*classification*.h5);;All .h5 files (*.h5)",
        )
        log.debug("Hand labels file dialog: %s", path or "cancelled")
        if not path:
            return
        name = Path(path).name
        try:
            self._bouts = labelled_bouts(read_classification_h5(path))
        except Exception as exc:
            # Usually the wrong .h5: pose files share the extension. A plain
            # message helps more than the reader's traceback, which is logged.
            log.warning("Could not read hand labels from %s", path, exc_info=exc)
            hint = " It is a pose file." if "poser_coords" in name else ""
            self._status.append(
                f"{name} is not a classification file of hand-labelled bouts."
                f"{hint} Pick the one saved when the bouts were labelled, "
                "usually <video>_classification.h5."
            )
            return
        self._labels_label.setText(f"{name}: {len(self._bouts)} bouts.")
        log.info(
            "Loaded %d hand-labelled bouts from %s; classes %s",
            len(self._bouts),
            path,
            sorted({bout.name for bout in self._bouts}),
        )
        self._score_if_ready()

    def _on_names_clicked(self) -> None:
        # One file type per filter. The macOS dialog greyed out every .ckpt when
        # one filter mixed *.ckpt with *.yml and *.yaml.
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select class names",
            "",
            "Checkpoint (*.ckpt);;Decoder config (*.yml *.yaml);;All files (*)",
        )
        log.debug("Class names file dialog: %s", path or "cancelled")
        if not path:
            return
        path = Path(path)
        if path.suffix == ".ckpt":
            classes = self._classes_from_checkpoint(path)
            if classes is None:
                log.warning(
                    "%s stores no class names; pick the decoder_config.yml it was "
                    "trained with",
                    path,
                )
                self._status.append(
                    f"{path.name} stores no class names. Pick the "
                    "decoder_config.yml it was trained with instead."
                )
                return
        else:
            try:
                classes = class_names_from_config(path)
            except Exception as exc:
                self._show_error(f"Could not read class names from {path.name}", exc)
                return
        log.info("Read class names from %s", path)
        self._classes_browsed = True
        self._set_classes(classes, path.name)
        self._score_if_ready()

    def _set_classes(self, classes: ClassNames, source: str) -> None:
        self._classes = classes
        names = ", ".join(classes.names[i] for i in sorted(classes.names))
        ignored = f" (ignores {', '.join(sorted(classes.ignored))})"
        self._names_label.setText(
            f"{source}: {names}{ignored if classes.ignored else ''}."
        )
        log.info(
            "Metrics class names set (%s): %s, ignored %s",
            source,
            classes.names,
            sorted(classes.ignored),
        )

    def _on_score_clicked(self) -> None:
        missing = [
            name
            for name, value in (
                ("predictions", self._predictions),
                ("hand labels", self._bouts),
                ("class names", self._classes),
            )
            if value is None
        ]
        log.debug("Score clicked; missing inputs: %s", missing)
        if missing:
            log.warning("Cannot score: load %s first", ", ".join(missing))
            self._status.append(f"Load {', '.join(missing)} first.")
            return
        self._score_now()

    def _score_if_ready(self) -> None:
        """Score as soon as all three inputs are there, so new predictions update."""
        inputs = (self._predictions, self._bouts, self._classes)
        log.debug(
            "Score if ready: predictions %s, hand labels %s, class names %s",
            self._predictions is not None,
            self._bouts is not None,
            self._classes is not None,
        )
        if all(value is not None for value in inputs):
            self._score_now()

    def _score_now(self) -> None:
        log.debug(
            "Scoring %d frames of predictions against %d hand-labelled bouts; "
            "class names %s, ignored %s",
            self._predictions.size,
            len(self._bouts),
            self._classes.names,
            sorted(self._classes.ignored),
        )
        unknown = sorted(
            {bout.name for bout in self._bouts}
            - set(self._classes.names.values())
            - self._classes.ignored
        )
        if unknown:
            log.warning(
                "Hand-labelled classes %s are not among the class names %s; "
                "their bouts are skipped",
                unknown,
                sorted(self._classes.names.values()),
            )
        last_stop = max((bout.stop for bout in self._bouts), default=0)
        if last_stop > self._predictions.size:
            log.warning(
                "Predictions cover %d frames but the last hand-labelled bout ends "
                "at frame %d",
                self._predictions.size,
                last_stop,
            )
        try:
            score = score_bouts(self._predictions, self._bouts, self._classes)
        except ValueError as exc:
            log.exception("Could not score the predictions")
            self._status.append(str(exc))
            return
        self._score = score
        self._summary.setText(self._summary_text(score))
        self._fill_table(score)
        self._draw_matrix()
        self._export_button.setEnabled(True)
        self._status.append(f"Scored. {self._summary_text(score)}")
        log.info(
            "Scored %d bouts: accuracy %.3f, balanced accuracy %.3f "
            "(%d ignored, %d skipped)",
            score.n_scored,
            score.accuracy,
            score.balanced_accuracy,
            score.n_ignored,
            score.n_skipped,
        )

    def _summary_text(self, score: DecoderScore) -> str:
        parts = [f"{score.n_scored} bouts scored"]
        if score.n_ignored:
            ignored = ", ".join(sorted(self._classes.ignored))
            parts.append(f"{score.n_ignored} ignored ({ignored})")
        if score.n_skipped:
            parts.append(f"{score.n_skipped} skipped")
        return (
            f"{', '.join(parts)}: accuracy {score.accuracy:.3f}, "
            f"balanced accuracy {score.balanced_accuracy:.3f}"
        )

    def _fill_table(self, score: DecoderScore) -> None:
        self._table.setRowCount(len(score.class_names))
        for row, name in enumerate(score.class_names):
            values = score.per_class[name]
            cells = (
                name,
                f"{values['precision']:.3f}",
                f"{values['recall']:.3f}",
                f"{values['f1-score']:.3f}",
                f"{int(values['support'])}",
            )
            for column, text in enumerate(cells):
                self._table.setItem(row, column, QTableWidgetItem(text))

    def _on_view_changed(self) -> None:
        log.debug(
            "Matrix view set to %s; a score to redraw: %s",
            self._view_combo.currentText(),
            self._score is not None,
        )
        if self._score is not None:
            self._draw_matrix()

    def _draw_matrix(self) -> None:
        """Rows are true classes, columns predicted, coloured by share of the row."""
        score = self._score
        counts = score.confusion_matrix.astype(float)
        row_totals = counts.sum(axis=1, keepdims=True)
        share = np.divide(
            counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0
        )
        show_percent = self._view_combo.currentText() == _PERCENT

        self._figure.clear()
        ax = self._figure.add_subplot(1, 1, 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(score.class_names)):
            for j in range(len(score.class_names)):
                if show_percent:
                    text = f"{100 * share[i, j]:.0f}%"
                else:
                    text = f"{int(counts[i, j])}"
                ax.text(
                    j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.5 else "black",
                )
        ticks = range(len(score.class_names))
        ax.set_xticks(ticks, score.class_names, color="#cccccc", fontsize=8)
        ax.set_yticks(ticks, score.class_names, color="#cccccc", fontsize=8)
        ax.set_xlabel("Predicted", color="#cccccc", fontsize=9)
        ax.set_ylabel("True (hand label)", color="#cccccc", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
        self._figure.tight_layout()
        self._canvas.draw()

    def _on_export_clicked(self) -> None:
        default = resolve_output_dir("metrics") / "confusion_matrix.png"
        if self._predictions_path is not None:
            stem = self._predictions_path.with_suffix("").name
            default = default.with_name(f"{stem}_confusion_matrix.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export confusion matrix", str(default), "PNG image (*.png)"
        )
        log.debug("Export dialog (default %s): %s", default, path or "cancelled")
        if not path:
            return
        path = Path(path).with_suffix(".png")
        try:
            self._figure.savefig(path, dpi=150, facecolor=self._figure.get_facecolor())
            written = [path, *save_score_csv(self._score, path)]
        except OSError as exc:
            self._show_error("Could not export", exc)
            return
        log.info(
            "Exported the confusion matrix and scores to %s",
            ", ".join(str(p) for p in written),
        )
        self._status.append("Exported " + ", ".join(p.name for p in written) + ".")

    def _show_error(self, message: str, exc: Exception) -> None:
        log.error(message, exc_info=exc)
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._status.append(f"ERROR: {message}: {exc}\n{details}")
