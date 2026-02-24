"""
_panels/data_panel.py
~~~~~~~~~~~~~~~~~~~~~
Data Panel — session management and file loading.

Provides:
* Drag-and-drop / browse for multiple pose files and videos (Feature 1)
* A session table showing all loaded files with their status
* Per-file activation (hides/shows napari layers for the active entry)
* Save / resume session functionality (Feature 4 / :class:`SessionManager`)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from poser.core.session import SessionManager, SessionEntry

if TYPE_CHECKING:
    import napari


class DataPanel(QWidget):
    """Napari dock widget for session / file management.

    Parameters
    ----------
    viewer:
        The running ``napari.Viewer`` instance.
    session:
        A shared :class:`~poser.core.session.SessionManager` instance.
        If ``None`` a new one is created.
    """

    def __init__(
        self,
        viewer: "napari.Viewer",
        session: Optional[SessionManager] = None,
    ):
        super().__init__()
        self._viewer = viewer
        self._session: SessionManager = session or SessionManager(viewer)

        self._build_ui()
        self._refresh_table()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── File loading group ─────────────────────────────────────────
        load_grp = QGroupBox("Load Files")
        load_lay = QVBoxLayout(load_grp)

        btn_add_pose = QPushButton("Add Pose File(s)…")
        btn_add_pose.setToolTip("Load DLC / SLEAP / PoseR HDF5 files.")
        btn_add_pose.clicked.connect(self._on_add_pose)
        load_lay.addWidget(btn_add_pose)

        btn_add_batch = QPushButton("Add Folder of Pose Files…")
        btn_add_batch.clicked.connect(self._on_add_folder)
        load_lay.addWidget(btn_add_batch)

        root.addWidget(load_grp)

        # ── Session table ──────────────────────────────────────────────
        session_grp = QGroupBox("Session")
        session_lay = QVBoxLayout(session_grp)

        self._session_list = QListWidget()
        self._session_list.setMinimumHeight(140)
        self._session_list.itemDoubleClicked.connect(self._on_activate)
        session_lay.addWidget(self._session_list)

        # Status key
        key_label = QLabel("🟢 done   🟡 active   ⚪ pending")
        key_label.setStyleSheet("font-size: 10px; color: grey;")
        session_lay.addWidget(key_label)

        btn_row = QHBoxLayout()
        btn_activate = QPushButton("Activate")
        btn_activate.setToolTip("Switch to the selected file.")
        btn_activate.clicked.connect(self._on_activate)

        btn_mark_done = QPushButton("Mark Done")
        btn_mark_done.clicked.connect(self._on_mark_done)

        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(self._on_remove)

        btn_next = QPushButton("Next Pending ▶")
        btn_next.setToolTip("Jump to the next un-annotated file.")
        btn_next.clicked.connect(self._on_next_pending)

        for btn in (btn_activate, btn_mark_done, btn_remove, btn_next):
            btn_row.addWidget(btn)
        session_lay.addLayout(btn_row)

        root.addWidget(session_grp)

        # ── Persistence ───────────────────────────────────────────────
        persist_grp = QGroupBox("Session File")
        persist_lay = QHBoxLayout(persist_grp)

        btn_save = QPushButton("Save Session")
        btn_save.clicked.connect(self._on_save_session)
        btn_load = QPushButton("Load Session")
        btn_load.clicked.connect(self._on_load_session)

        persist_lay.addWidget(btn_save)
        persist_lay.addWidget(btn_load)
        root.addWidget(persist_grp)

        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_add_pose(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select pose files", "",
            "HDF5 / CSV files (*.h5 *.hdf5 *.csv);;All files (*)"
        )
        if not files:
            return
        self._session.add_batch([Path(f) for f in files])
        self._refresh_table()

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if not folder:
            return
        folder_path = Path(folder)
        pose_files = (
            list(folder_path.glob("**/*.h5"))
            + list(folder_path.glob("**/*.hdf5"))
            + list(folder_path.glob("**/*.csv"))
        )
        if not pose_files:
            QMessageBox.information(self, "No files", "No .h5/.hdf5/.csv files found.")
            return
        self._session.add_batch(pose_files)
        self._refresh_table()

    def _on_activate(self, _item=None) -> None:
        idx = self._session_list.currentRow()
        if idx < 0:
            return
        self._session.activate(idx)
        self._refresh_table()

    def _on_mark_done(self) -> None:
        idx = self._session_list.currentRow()
        if idx < 0:
            return
        self._session.mark_done(idx)
        self._refresh_table()

    def _on_remove(self) -> None:
        idx = self._session_list.currentRow()
        if idx < 0:
            return
        self._session.remove(idx)
        self._refresh_table()

    def _on_next_pending(self) -> None:
        next_idx = self._session.next_pending()
        if next_idx is None:
            QMessageBox.information(self, "Session complete", "All files have been processed!")
            return
        self._session.activate(next_idx)
        self._refresh_table()

    def _on_save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "session.json", "JSON (*.json)"
        )
        if path:
            self._session.save(Path(path))

    def _on_load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load session", "", "JSON (*.json)"
        )
        if path:
            self._session.load(Path(path))
            self._refresh_table()

    # ------------------------------------------------------------------
    # Table refresh
    # ------------------------------------------------------------------

    def _refresh_table(self) -> None:
        self._session_list.clear()
        for i, entry in enumerate(self._session.entries):
            icon = {"done": "🟢", "active": "🟡", "pending": "⚪"}.get(entry.status, "⚪")
            text = f"{icon}  {entry.stem}"
            if entry.video_path:
                text += f"  [{entry.video_path.name}]"
            item = QListWidgetItem(text)
            if entry.status == "active":
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._session_list.addItem(item)
