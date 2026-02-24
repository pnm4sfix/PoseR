"""
_panels/train_panel.py
~~~~~~~~~~~~~~~~~~~~~~
Train Panel — launches model training via the CLI in a background subprocess.

The training process runs completely outside the napari event loop, keeping
the GUI fully responsive.  A live log readout streams subprocess stdout.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from poser.core.session import SessionManager

if TYPE_CHECKING:
    import napari


class TrainPanel(QWidget):
    """Napari dock widget for launching behaviour decoder training.

    Parameters
    ----------
    viewer:
        The running ``napari.Viewer`` instance.
    session:
        Shared :class:`~poser.core.session.SessionManager`.
    """

    _log_signal = Signal(str)
    _done_signal = Signal(int)   # exit code

    def __init__(
        self,
        viewer: "napari.Viewer",
        session: SessionManager,
    ):
        super().__init__()
        self._viewer = viewer
        self._session = session
        self._training_proc: Optional[subprocess.Popen] = None
        self._training_files: List[Path] = []
        self._config_path: Optional[Path] = None

        self._build_ui()
        self._log_signal.connect(self._on_log)
        self._done_signal.connect(self._on_training_done)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Training data ─────────────────────────────────────────────
        data_grp = QGroupBox("Training Data")
        data_lay = QVBoxLayout(data_grp)

        btn_add_files = QPushButton("Add labelled HDF5 files…")
        btn_add_files.clicked.connect(self._on_add_files)
        data_lay.addWidget(btn_add_files)

        btn_use_session = QPushButton("Use session files (done only)")
        btn_use_session.clicked.connect(self._on_use_session_files)
        data_lay.addWidget(btn_use_session)

        self._file_list_label = QLabel("No files selected.")
        self._file_list_label.setStyleSheet("font-size: 10px; color: grey;")
        self._file_list_label.setWordWrap(True)
        data_lay.addWidget(self._file_list_label)

        root.addWidget(data_grp)

        # ── Config ────────────────────────────────────────────────────
        cfg_grp = QGroupBox("Training Config")
        cfg_lay = QVBoxLayout(cfg_grp)

        btn_browse_cfg = QPushButton("Browse config YAML…")
        btn_browse_cfg.clicked.connect(self._on_browse_config)
        cfg_lay.addWidget(btn_browse_cfg)

        self._cfg_label = QLabel("Using defaults (no config loaded).")
        self._cfg_label.setStyleSheet("font-size: 10px; color: grey;")
        self._cfg_label.setWordWrap(True)
        cfg_lay.addWidget(self._cfg_label)

        # Quick override: epochs
        epochs_row = QHBoxLayout()
        epochs_row.addWidget(QLabel("Max epochs:"))
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 10000)
        self._epochs_spin.setValue(200)
        epochs_row.addWidget(self._epochs_spin)
        cfg_lay.addLayout(epochs_row)

        # Run name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Run name:"))
        self._run_name_edit = QLineEdit("run_01")
        name_row.addWidget(self._run_name_edit)
        cfg_lay.addLayout(name_row)

        self._auto_lr_cb = QCheckBox("Auto LR-find before training")
        cfg_lay.addWidget(self._auto_lr_cb)

        root.addWidget(cfg_grp)

        # ── Controls ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_train = QPushButton("▶  Start Training")
        self._btn_train.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self._btn_train.clicked.connect(self._on_start_training)
        btn_row.addWidget(self._btn_train)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop_training)
        btn_row.addWidget(self._btn_stop)

        root.addLayout(btn_row)

        # ── Log ───────────────────────────────────────────────────────
        log_grp = QGroupBox("Training Log")
        log_lay = QVBoxLayout(log_grp)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFixedHeight(200)
        log_lay.addWidget(self._log_text)

        root.addWidget(log_grp)
        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select labelled HDF5 files", "",
            "HDF5 files (*.h5 *.hdf5);;All files (*)"
        )
        if files:
            self._training_files = [Path(f) for f in files]
            self._update_file_label()

    def _on_use_session_files(self) -> None:
        done = [e.pose_path for e in self._session.entries if e.status == "done"]
        if not done:
            QMessageBox.information(self, "No done files",
                                    "Mark files as 'done' in the Data panel first.")
            return
        self._training_files = done
        self._update_file_label()

    def _on_browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select config YAML", "", "YAML (*.yml *.yaml)"
        )
        if path:
            self._config_path = Path(path)
            self._cfg_label.setText(str(self._config_path))

    def _on_start_training(self) -> None:
        if not self._training_files:
            QMessageBox.warning(self, "No files", "Add labelled HDF5 files first.")
            return
        if self._training_proc is not None and self._training_proc.poll() is None:
            QMessageBox.warning(self, "Already running", "Training is already in progress.")
            return

        cmd = [
            sys.executable, "-m", "poser.cli.main", "train",
            *[str(f) for f in self._training_files],
            "--epochs", str(self._epochs_spin.value()),
            "--name", self._run_name_edit.text(),
        ]
        if self._config_path:
            cmd += ["--config", str(self._config_path)]
        if self._auto_lr_cb.isChecked():
            cmd += ["--auto-lr"]

        self._log_text.clear()
        self._log_signal.emit(f"$ {' '.join(cmd)}\n")

        try:
            self._training_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Launch error", str(exc))
            return

        self._btn_train.setEnabled(False)
        self._btn_stop.setEnabled(True)

        # Stream stdout in background thread
        threading.Thread(target=self._stream_output, daemon=True).start()

    def _on_stop_training(self) -> None:
        if self._training_proc and self._training_proc.poll() is None:
            self._training_proc.terminate()
            self._log_signal.emit("\n[TRAINING STOPPED BY USER]")
        self._btn_train.setEnabled(True)
        self._btn_stop.setEnabled(False)

    def _on_log(self, text: str) -> None:
        self._log_text.append(text.rstrip())

    def _on_training_done(self, exit_code: int) -> None:
        self._btn_train.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if exit_code == 0:
            self._log_signal.emit("\n✅ Training completed successfully.")
        else:
            self._log_signal.emit(f"\n❌ Training exited with code {exit_code}.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stream_output(self) -> None:
        """Read subprocess stdout line-by-line and emit to log."""
        proc = self._training_proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._log_signal.emit(line)
        proc.wait()
        self._done_signal.emit(proc.returncode)

    def _update_file_label(self) -> None:
        n = len(self._training_files)
        names = ", ".join(f.name for f in self._training_files[:3])
        suffix = f" … (+{n-3} more)" if n > 3 else ""
        self._file_list_label.setText(f"{n} file(s): {names}{suffix}")
