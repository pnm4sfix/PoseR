"""
_panels/ethogram_panel.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Ethogram Panel — colour-coded per-frame behaviour visualisation.

Displays two matplotlib rows embedded in a Qt widget (no napari-plot / OpenGL):
  * Annotations row — ground-truth labels from a *_labels.npy file
  * Predictions row — per-frame inference output

A white vertical cursor line tracks the active napari viewer frame.
The ``predictions_ready`` signal from InferencePanel auto-populates the
Predictions row when both panels are open.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import json
import numpy as np
import napari
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Class colour palette (up to 20 classes, cycling)
# ---------------------------------------------------------------------------
_CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
]


# ---------------------------------------------------------------------------
# Run-length encoding helper
# ---------------------------------------------------------------------------

def _rle(labels: np.ndarray):
    """Return (starts, ends, values) via run-length encoding."""
    if len(labels) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    change = np.concatenate(([True], labels[1:] != labels[:-1], [True]))
    starts = np.where(change[:-1])[0]
    ends   = np.where(change[1:])[0] + 1
    values = labels[starts]
    return starts, ends, values


# ---------------------------------------------------------------------------
# EthogramPanel
# ---------------------------------------------------------------------------

class EthogramPanel(QWidget):
    """Napari dock widget showing a colour-coded behaviour ethogram.

    Two horizontal bar tracks rendered with matplotlib:
      - top row: annotation labels  (*_labels.npy)
      - bottom row: predicted labels (*_predictions.npy)

    A white vertical line tracks the current napari frame.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        label_names: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self._viewer = viewer
        self._label_names: Dict[int, str] = label_names or {}

        self._annotations: Optional[np.ndarray] = None
        self._predictions: Optional[np.ndarray] = None
        self._cursor_lines: list = []   # matplotlib Line2D references

        self._canvas = None             # FigureCanvasQTAgg — set in _build_ui
        self._ax_ann = None
        self._ax_pred = None

        self._build_ui()
        self._viewer.dims.events.current_step.connect(self._on_frame_changed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Load controls ──────────────────────────────────────────────
        ctrl_grp = QGroupBox("Load Labels")
        ctrl_lay = QVBoxLayout(ctrl_grp)

        ann_row = QHBoxLayout()
        ann_row.addWidget(QLabel("Annotations:"))
        self._ann_label = QLabel("None")
        self._ann_label.setStyleSheet("font-size: 10px; color: grey;")
        self._ann_label.setWordWrap(True)
        ann_row.addWidget(self._ann_label, 1)
        btn_ann = QPushButton("Browse…")
        btn_ann.setFixedWidth(70)
        btn_ann.clicked.connect(self._on_browse_annotations)
        ann_row.addWidget(btn_ann)
        ctrl_lay.addLayout(ann_row)

        pred_row = QHBoxLayout()
        pred_row.addWidget(QLabel("Predictions:"))
        self._pred_label = QLabel("None")
        self._pred_label.setStyleSheet("font-size: 10px; color: grey;")
        self._pred_label.setWordWrap(True)
        pred_row.addWidget(self._pred_label, 1)
        btn_pred = QPushButton("Browse…")
        btn_pred.setFixedWidth(70)
        btn_pred.clicked.connect(self._on_browse_predictions)
        pred_row.addWidget(btn_pred)
        ctrl_lay.addLayout(pred_row)

        btn_refresh = QPushButton("Refresh Ethogram")
        btn_refresh.clicked.connect(self._refresh)
        ctrl_lay.addWidget(btn_refresh)

        root.addWidget(ctrl_grp)

        # ── Legend ─────────────────────────────────────────────────────
        self._legend_widget = _LegendWidget(self._label_names)
        root.addWidget(self._legend_widget)

        # ── Matplotlib canvas ─────────────────────────────────────────
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            fig = Figure(facecolor="#1e1e1e")
            canvas = FigureCanvasQTAgg(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.setMinimumHeight(130)

            self._fig = fig
            self._canvas = canvas

            self._ax_ann  = fig.add_subplot(2, 1, 1)
            self._ax_pred = fig.add_subplot(2, 1, 2, sharex=self._ax_ann)
            self._style_axes()

            root.addWidget(canvas, 1)

        except Exception as exc:
            err_lbl = QLabel(f"matplotlib canvas unavailable: {exc}")
            err_lbl.setStyleSheet("color: orange; font-size: 10px;")
            root.addWidget(err_lbl)

        # ── Live frame status strip ────────────────────────────────────
        self._frame_status = QLabel("Frame —")
        self._frame_status.setStyleSheet(
            "font-size: 11px; font-family: monospace; "
            "background: #1a1a1a; color: #dddddd; "
            "padding: 3px 6px; border-radius: 3px;"
        )
        self._frame_status.setWordWrap(True)
        root.addWidget(self._frame_status)

    def _style_axes(self) -> None:
        """Apply dark-theme styling to both axes."""
        for ax, row_label in [
            (self._ax_ann,  "Annotations"),
            (self._ax_pred, "Predictions"),
        ]:
            ax.set_facecolor("#2d2d2d")
            ax.set_ylim(0, 1)
            ax.set_yticks([0.5])
            ax.set_yticklabels([row_label], color="#cccccc", fontsize=8)
            ax.tick_params(axis="x", colors="#cccccc", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")

        # Only bottom axis shows x tick labels
        self._ax_ann.tick_params(labelbottom=False)
        self._ax_pred.set_xlabel("Frame", color="#cccccc", fontsize=8)
        self._fig.subplots_adjust(
            left=0.12, right=0.99, top=0.97, bottom=0.18, hspace=0.08
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_predictions(
        self,
        predictions: np.ndarray,
        checkpoint: Optional[Path] = None,
    ) -> None:
        """Called by InferencePanel.predictions_ready signal."""
        self._predictions = predictions.astype(np.int64)
        name = checkpoint.name if checkpoint is not None else "inference"
        self._pred_label.setText(name)
        # Try to pull class names out of the checkpoint
        if checkpoint is not None:
            try:
                import torch
                raw = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
                hp = raw.get("hyper_parameters", {})
                cn = hp.get("class_names") or hp.get("data_cfg", {}).get("class_names")
                if cn and isinstance(cn, dict):
                    self._label_names = {int(k): v for k, v in cn.items()}
                elif cn and isinstance(cn, (list, tuple)):
                    self._label_names = {i: v for i, v in enumerate(cn)}
            except Exception:
                pass
        self._refresh()

    def load_annotations(self, annotations: np.ndarray) -> None:
        self._annotations = annotations.astype(np.int64)
        self._refresh()

    def set_label_names(self, label_names: Dict[int, str]) -> None:
        self._label_names = label_names
        self._legend_widget.update_names(label_names)
        self._refresh()

    # ------------------------------------------------------------------
    # Browse slots
    # ------------------------------------------------------------------

    def _on_browse_annotations(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select *_labels.npy", "",
            "NumPy arrays (*.npy);;All files (*)"
        )
        if path:
            self._annotations = np.load(path).astype(np.int64)
            self._ann_label.setText(Path(path).name)
            self._try_load_schema(path)
            self._refresh()

    def _on_browse_predictions(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select *_predictions.npy", "",
            "NumPy arrays (*.npy);;All files (*)"
        )
        if path:
            self._predictions = np.load(path).astype(np.int64)
            self._pred_label.setText(Path(path).name)
            self._try_load_schema(path)
            self._refresh()

    def _try_load_schema(self, npy_path: str) -> None:
        """Look for a sibling *_schema.json and load class names from it."""
        p = Path(npy_path)
        # Strip common suffixes to find the base stem
        for suffix in ("_labels", "_predictions", "_pose"):
            if p.stem.endswith(suffix):
                base = p.parent / (p.stem[: -len(suffix)] + "_schema.json")
                if base.exists():
                    try:
                        with open(base) as fh:
                            raw = json.load(fh)
                        self._label_names = {int(k): v for k, v in raw.items()}
                        self._legend_widget.update_names(self._label_names)
                    except Exception:
                        pass
                    return

    # ------------------------------------------------------------------
    # Redraw
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._canvas is None:
            return

        self._cursor_lines.clear()
        self._ax_ann.cla()
        self._ax_pred.cla()
        self._style_axes()

        T = 0
        if self._annotations is not None:
            T = max(T, len(self._annotations))
            self._draw_track(self._ax_ann, self._annotations)
        if self._predictions is not None:
            T = max(T, len(self._predictions))
            self._draw_track(self._ax_pred, self._predictions)

        if T > 0:
            self._ax_ann.set_xlim(0, T)

        # Cursor
        try:
            frame = float(self._viewer.dims.current_step[0])
        except Exception:
            frame = 0.0
        for ax in (self._ax_ann, self._ax_pred):
            line = ax.axvline(frame, color="white", linewidth=1.0, alpha=0.85, zorder=10)
            self._cursor_lines.append(line)

        self._canvas.draw()
        self._legend_widget.update_names(self._label_names)
        self._update_frame_status()  # populate status strip immediately

    def _draw_track(self, ax, labels: np.ndarray) -> None:
        """Draw one ethogram row using broken_barh (one call per class)."""
        if len(labels) == 0:
            return
        starts, ends, values = _rle(labels)

        # Group consecutive runs by class for efficient drawing
        class_runs: dict = defaultdict(list)
        for s, e, v in zip(starts, ends, values):
            class_runs[int(v)].append((int(s), int(e) - int(s)))

        for cls_id, runs in sorted(class_runs.items()):
            if cls_id < 0:
                color = "#888888"  # gray = unlabelled bout
            else:
                color = _CLASS_COLORS[cls_id % len(_CLASS_COLORS)]
            ax.broken_barh(runs, (0.05, 0.90), facecolors=color, edgecolors="none")

    # ------------------------------------------------------------------
    # Frame cursor sync + live status
    # ------------------------------------------------------------------

    def _label_name(self, cls_id: int) -> str:
        """Return display name for a class id, with color markup."""
        if cls_id < 0:
            return '<span style="color:#888888">unlabelled</span>'
        name = self._label_names.get(cls_id, f"cls{cls_id}")
        color = _CLASS_COLORS[cls_id % len(_CLASS_COLORS)]
        return f'<span style="color:{color}; font-weight:bold">{name}</span>'

    def _update_frame_status(self) -> None:
        try:
            frame = int(self._viewer.dims.current_step[0])
        except Exception:
            frame = 0

        parts = [f"Frame <b>{frame}</b>"]

        if self._annotations is not None and frame < len(self._annotations):
            parts.append(f"Ann: {self._label_name(int(self._annotations[frame]))}")
        else:
            parts.append("Ann: <span style='color:#555'>—</span>")

        if self._predictions is not None and frame < len(self._predictions):
            parts.append(f"Pred: {self._label_name(int(self._predictions[frame]))}")
        else:
            parts.append("Pred: <span style='color:#555'>—</span>")

        self._frame_status.setText("&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts))

    def _on_frame_changed(self, event=None) -> None:
        if self._canvas is None or not self._cursor_lines:
            return
        try:
            frame = float(self._viewer.dims.current_step[0])
            for line in self._cursor_lines:
                line.set_xdata([frame, frame])
            self._canvas.draw_idle()
            self._update_frame_status()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Legend widget
# ---------------------------------------------------------------------------

class _LegendWidget(QWidget):
    """Small row of colour swatches → class name labels."""

    def __init__(self, label_names: Dict[int, str]):
        super().__init__()
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(4)
        self.update_names(label_names)

    def update_names(self, label_names: Dict[int, str]) -> None:
        while self._lay.count():
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for cls_id in sorted(label_names.keys()):
            name = label_names[cls_id]
            color = _CLASS_COLORS[int(cls_id) % len(_CLASS_COLORS)]
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background-color: {color}; border: 1px solid #555;"
            )
            lbl = QLabel(f"{cls_id}: {name}")
            lbl.setStyleSheet("font-size: 10px;")
            self._lay.addWidget(swatch)
            self._lay.addWidget(lbl)

        self._lay.addStretch()
