"""
_panels/inference_panel.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Inference Panel — single-file and batch behaviour decoding.

Features:
* Load a model checkpoint (file-drop or browse)
* Run inference on the active session file
* Run batch inference on all session files  (Feature 1)
* Progress display for batch jobs
* Display predicted labels in a Shapes / Points layer overlay
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from poser.core.session import SessionManager

if TYPE_CHECKING:
    import napari


class InferencePanel(QWidget):
    """Napari dock widget for behaviour inference.

    Parameters
    ----------
    viewer:
        The running ``napari.Viewer`` instance.
    session:
        Shared :class:`~poser.core.session.SessionManager`.
    bouts_getter:
        Callable returning the current bout list from the Analysis panel.
    """

    # Custom Qt signals to safely update UI from background thread
    _progress_signal = Signal(int)
    _status_signal = Signal(str)

    def __init__(
        self,
        viewer: "napari.Viewer",
        session: SessionManager,
        bouts_getter=None,
    ):
        super().__init__()
        self._viewer = viewer
        self._session = session
        self._bouts_getter = bouts_getter
        self._checkpoint: Optional[Path] = None
        self._predictions: Dict[int, int] = {}

        self._build_ui()
        self._progress_signal.connect(self._on_progress)
        self._status_signal.connect(self._on_status)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Checkpoint selection ──────────────────────────────────────
        ckpt_grp = QGroupBox("Model Checkpoint")
        ckpt_lay = QVBoxLayout(ckpt_grp)

        self._ckpt_label = QLabel("No checkpoint loaded.")
        self._ckpt_label.setStyleSheet("font-size: 10px; color: grey;")
        self._ckpt_label.setWordWrap(True)
        ckpt_lay.addWidget(self._ckpt_label)

        btn_browse = QPushButton("Browse for checkpoint (.ckpt / .pt)…")
        btn_browse.clicked.connect(self._on_browse_checkpoint)
        ckpt_lay.addWidget(btn_browse)

        root.addWidget(ckpt_grp)

        # ── Single-file inference ─────────────────────────────────────
        single_grp = QGroupBox("Single File Inference")
        single_lay = QVBoxLayout(single_grp)

        btn_predict = QPushButton("▶  Predict Behaviours (active file)")
        btn_predict.clicked.connect(self._on_predict_single)
        single_lay.addWidget(btn_predict)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        single_lay.addWidget(self._result_label)

        root.addWidget(single_grp)

        # ── Batch inference ───────────────────────────────────────────
        batch_grp = QGroupBox("Batch Inference (all session files)")
        batch_lay = QVBoxLayout(batch_grp)

        btn_batch = QPushButton("▶▶  Run Batch Inference")
        btn_batch.clicked.connect(self._on_predict_batch)
        batch_lay.addWidget(btn_batch)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        batch_lay.addWidget(self._progress_bar)

        self._status_text = QTextEdit()
        self._status_text.setReadOnly(True)
        self._status_text.setFixedHeight(80)
        batch_lay.addWidget(self._status_text)

        root.addWidget(batch_grp)

        # ── Export ────────────────────────────────────────────────────
        export_grp = QGroupBox("Export Predictions")
        export_lay = QHBoxLayout(export_grp)
        btn_export = QPushButton("Save Predictions (JSON)…")
        btn_export.clicked.connect(self._on_export_predictions)
        export_lay.addWidget(btn_export)
        root.addWidget(export_grp)

        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "",
            "Checkpoints (*.ckpt *.pt);;All files (*)"
        )
        if path:
            self._checkpoint = Path(path)
            self._ckpt_label.setText(str(self._checkpoint))

    def _on_predict_single(self) -> None:
        if not self._checkpoint:
            QMessageBox.warning(self, "No checkpoint", "Please select a model checkpoint first.")
            return

        entry = self._session.active_entry
        if entry is None or entry.coords_data is None:
            QMessageBox.warning(self, "No data", "No pose data loaded in the active session entry.")
            return

        bouts = self._bouts_getter() if self._bouts_getter else []
        if not bouts:
            QMessageBox.warning(self, "No bouts", "Run bout detection first.")
            return

        from poser.api import PoseR
        try:
            poser = PoseR()
            X = poser.preprocess(entry.coords_data, bouts)
            self._predict_from_array(X, bouts)
        except Exception as exc:
            QMessageBox.critical(self, "Inference error", str(exc))
            return

        self._result_label.setText(
            f"Predicted {len(self._predictions)} bouts. "
            f"Labels: {list(self._predictions.values())}"
        )

    def _on_predict_batch(self) -> None:
        if not self._checkpoint:
            QMessageBox.warning(self, "No checkpoint", "Please select a model checkpoint first.")
            return

        n = len(self._session.entries)
        if n == 0:
            QMessageBox.warning(self, "Empty session", "Add files to the session first.")
            return

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_text.clear()

        def run():
            from poser.core.batch import BatchJob
            from poser.api import PoseR

            poser = PoseR()

            def cb(done, total, path):
                self._progress_signal.emit(int(100 * done / total))
                self._status_signal.emit(f"[{done}/{total}] {path}")

            job = BatchJob(
                pose_files=[e.pose_path for e in self._session.entries],
                mode="behaviour",
                checkpoint=self._checkpoint,
                config=poser.config,
                output_dir=Path("batch_output"),
                n_individuals=1,
                progress_callback=cb,
            )
            results = job.run()
            n_ok = sum(1 for r in results if r.success)
            self._status_signal.emit(f"Done: {n_ok}/{len(results)} succeeded.")
            self._progress_signal.emit(100)

        threading.Thread(target=run, daemon=True).start()

    def _on_export_predictions(self) -> None:
        import json

        if not self._predictions:
            QMessageBox.warning(self, "No predictions", "Run inference first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save predictions", "predictions.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w") as fh:
                json.dump({str(k): int(v) for k, v in self._predictions.items()}, fh, indent=2)

    # ------------------------------------------------------------------
    # Signal receivers
    # ------------------------------------------------------------------

    def _on_progress(self, value: int) -> None:
        self._progress_bar.setValue(value)

    def _on_status(self, text: str) -> None:
        self._status_text.append(text)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predict_from_array(self, X: np.ndarray, bouts: List[dict]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from poser.models.registry import load_model
        from poser.api import PoseR

        poser = PoseR()
        mcfg = poser.config.model

        model = load_model(
            mcfg.architecture,
            checkpoint=self._checkpoint,
            num_class=mcfg.num_class,
            num_nodes=mcfg.num_nodes,
            in_channels=mcfg.in_channels,
            layout=mcfg.layout,
        )
        model.eval()

        tensor = torch.from_numpy(X).float()
        dl = DataLoader(TensorDataset(tensor), batch_size=32, shuffle=False)

        all_preds: List[int] = []
        with torch.no_grad():
            for (batch,) in dl:
                logits = model(batch)
                all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

        self._predictions = {b["start"]: p for b, p in zip(bouts, all_preds)}
