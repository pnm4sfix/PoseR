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
from typing import List, Optional

import napari
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


# ---------------------------------------------------------------------------
# Pose-file → napari points helpers  (mirrors original _widget.py logic)
# ---------------------------------------------------------------------------

def _parse_dlc_df(dlc_data):
    """Parse a DLC DataFrame (from pd.read_hdf / pd.read_csv) into coords_data.

    Returns dict:  individual_name → {"x": DataFrame, "y": DataFrame, "ci": DataFrame}
    where each DataFrame is shape (n_bodyparts, n_frames) with frame-index columns.
    Mirrors widget.read_dlc_h5.
    """
    import pandas as pd

    data_t = dlc_data.transpose()
    try:
        _ = data_t["individuals"]
        data_t = data_t.reset_index()
    except (KeyError, TypeError):
        data_t["individuals"] = ["individual1"] * data_t.shape[0]
        data_t = (
            data_t.reset_index()
            .set_index(["scorer", "individuals", "bodyparts", "coords"])
            .reset_index()
        )

    coords_data = {}
    for individual in data_t.individuals.unique():
        indv1 = data_t[data_t.individuals == individual].copy()
        # Columns after reset_index are strings / ints for frame numbers;
        # the original widget uses `0:` slice (selects columns >= 0 i.e. frame cols)
        x = indv1.loc[indv1.coords == "x", 0:].reset_index(drop=True)
        y = indv1.loc[indv1.coords == "y", 0:].reset_index(drop=True)
        ci = indv1.loc[indv1.coords == "likelihood", 0:].reset_index(drop=True)
        if x.empty:
            continue
        coords_data[individual] = {"x": x, "y": y, "ci": ci}
    return coords_data


def _coords_data_to_points(coords_data):
    """Convert coords_data dict to (points, properties) for a napari Points layer.

    Mirrors widget.get_points(): z = frame index tiled across body-part rows.
    """
    import pandas as pd

    all_pts, all_conf, all_ind, all_node = [], [], [], []
    for ind_i, (_, indv) in enumerate(coords_data.items()):
        x = indv["x"]
        y = indv["y"]
        ci = indv["ci"]
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x)
            y = pd.DataFrame(y)
            ci = pd.DataFrame(ci)
        x_flat = x.to_numpy().flatten().astype(float)
        y_flat = y.to_numpy().flatten().astype(float)
        ci_flat = ci.to_numpy().flatten().astype(float) if ci is not None else np.zeros_like(x_flat)
        # Mirror get_points: frame numbers are column labels, tile across nodes
        z_flat = np.tile(x.columns.to_numpy(), x.shape[0]).astype(float)
        n_nodes, n_frames = x.shape
        node_flat = np.repeat(np.arange(n_nodes), n_frames)
        pts = np.column_stack([z_flat, y_flat, x_flat])
        # Drop ghost slots: undetected frames have NaN or zero in all of x, y, ci
        nan_mask = np.isnan(x_flat) | np.isnan(y_flat)
        zero_mask = (x_flat == 0) & (y_flat == 0) & (ci_flat == 0)
        keep = ~(nan_mask | zero_mask)
        all_pts.append(pts[keep])
        all_conf.append(ci_flat[keep])
        all_ind.append(np.full(keep.sum(), ind_i, dtype=int))
        all_node.append(node_flat[keep])
    points = np.vstack(all_pts)
    return points, {
        "confidence": np.concatenate(all_conf).astype(float),
        "ind": np.concatenate(all_ind),
        "node": np.concatenate(all_node),
    }


def _load_pose_as_points(path: str):
    """Parse a DLC / SLEAP / PoseR pose file into napari-ready arrays.

    Returns
    -------
    points : ndarray, shape (N, 3)  — columns (frame, y, x)
    properties : dict — keys: confidence, ind, node
    coords_data : dict — raw per-individual DataFrames, for Analysis Panel
    """
    import pandas as pd

    path = str(path)

    # ── 1. PoseR pytables format (filename contains "poser", or fallback try) ──
    try:
        import tables as tb  # type: ignore
        coords_data = {}
        with tb.open_file(path, mode="r") as coords_file:
            for ind in coords_file.list_nodes("/"):
                arr = ind.coords[:]  # shape (3, n_nodes, n_frames): [0]=x,[1]=y,[2]=ci
                coords_data[ind._v_name] = {
                    "x": pd.DataFrame(arr[0]),
                    "y": pd.DataFrame(arr[1]),
                    "ci": pd.DataFrame(arr[2]),
                }
        if coords_data:
            points, props = _coords_data_to_points(coords_data)
            return points, props, coords_data
    except Exception:
        pass

    # ── 2. DLC h5 format ──────────────────────────────────────────────────────
    if ".h5" in path or ".hdf5" in path:
        try:
            dlc_data = pd.read_hdf(path)
            coords_data = _parse_dlc_df(dlc_data)
            if coords_data:
                points, props = _coords_data_to_points(coords_data)
                return points, props, coords_data
        except Exception:
            pass

    # ── 3. DLC CSV format ─────────────────────────────────────────────────────
    if ".csv" in path:
        try:
            dlc_data = pd.read_csv(path, header=[0, 1, 2], index_col=0)
            coords_data = _parse_dlc_df(dlc_data)
            if coords_data:
                points, props = _coords_data_to_points(coords_data)
                return points, props, coords_data
        except Exception:
            pass

    # ── 4. SLEAP analysis h5 format ───────────────────────────────────────────
    # Layout: /tracks (n_tracks, 2, n_nodes, n_frames) where axis-1 is [x, y]
    #         /point_scores (n_tracks, n_nodes, n_frames)
    if ".h5" in path or ".hdf5" in path:
        try:
            import pandas as pd
            import tables as tb  # type: ignore
            coords_data = {}
            with tb.open_file(path, mode="r") as f:
                node_names = list(f.root._v_children.keys())
                if "tracks" in node_names:
                    tracks = f.root.tracks[:]          # (n_tracks, 2, n_nodes, n_frames)
                    n_tracks, _, n_nodes, n_frames = tracks.shape
                    if "point_scores" in node_names:
                        scores = f.root.point_scores[:]  # (n_tracks, n_nodes, n_frames)
                    else:
                        scores = np.ones((n_tracks, n_nodes, n_frames), dtype=np.float32)
                    for t in range(n_tracks):
                        x_arr = tracks[t, 0, :, :].astype(float)  # (n_nodes, n_frames)
                        y_arr = tracks[t, 1, :, :].astype(float)
                        ci_arr = scores[t, :, :].astype(float)
                        ind_name = f"ind{t + 1}"
                        coords_data[ind_name] = {
                            "x": pd.DataFrame(x_arr),
                            "y": pd.DataFrame(y_arr),
                            "ci": pd.DataFrame(ci_arr),
                        }
            if coords_data:
                points, props = _coords_data_to_points(coords_data)
                return points, props, coords_data
        except Exception:
            pass

    raise ValueError(f"Could not parse pose file: {path}")


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
        viewer: napari.Viewer,
        session: Optional[SessionManager] = None,
    ):
        super().__init__()
        self._viewer = viewer
        self._session: SessionManager = session or SessionManager(viewer=viewer)

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

        btn_add_video = QPushButton("Add Video(s)…")
        btn_add_video.setToolTip("Load video files to run pose estimation on.")
        btn_add_video.clicked.connect(self._on_add_video)
        load_lay.addWidget(btn_add_video)

        btn_add_zarr = QPushButton("Add Zarr Video…")
        btn_add_zarr.setToolTip(
            "Load a .zarr array store as video.\n"
            "Expected shape: (frames, cameras, H, W, C) or (frames, H, W, C) or (frames, H, W).\n"
            "Multi-camera arrays are split into one Image layer per camera."
        )
        btn_add_zarr.clicked.connect(self._on_add_zarr)
        load_lay.addWidget(btn_add_zarr)

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
        self._session.add_batch([(Path(f),) for f in files])
        self._refresh_table()

    def _on_add_zarr(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select zarr store (.zarr folder)"
        )
        if not folder:
            return
        path = Path(folder)
        # Quick validation: must be openable as a zarr array
        try:
            import zarr  # type: ignore
            zarr.open_array(str(path), mode="r")
        except Exception as exc:
            QMessageBox.warning(self, "Invalid zarr", str(exc))
            return
        self._session.add_batch([("", path)])
        self._refresh_table()

    def _on_add_video(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select video files", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.tif *.tiff);;All files (*)"
        )
        if not files:
            return
        # Video-only entries: pose_path is empty, video_path is set.
        # Pose estimation can then be run from the Inference Panel.
        self._session.add_batch([("", Path(f)) for f in files])
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
        self._session.add_batch([(f,) for f in pose_files])
        self._refresh_table()

    def _on_activate(self, _item=None) -> None:
        idx = self._session_list.currentRow()
        if idx < 0:
            return
        entry = self._session.activate(idx)
        self._load_entry_layers(entry)
        self._refresh_table()

    def _load_entry_layers(self, entry) -> None:
        """Load video and/or pose layers for *entry* if not already present."""
        existing = {layer.name for layer in self._viewer.layers}

        # ── Video ────────────────────────────────────────────────────
        if entry.video_path:
            if str(entry.video_path).endswith(".zarr"):
                # ── Zarr branch ──────────────────────────────────────
                try:
                    import zarr          # type: ignore
                    import dask.array as da  # type: ignore

                    z = zarr.open_array(str(entry.video_path), mode="r")
                    zarr_da = da.from_zarr(z)
                    stem = Path(entry.video_path).stem

                    if zarr_da.ndim == 5:
                        # (frames, cameras, H, W, C)  — split by camera
                        n_cams = zarr_da.shape[1]
                        for cam in range(n_cams):
                            cam_name = f"{stem}_cam{cam}"
                            if cam_name not in existing:
                                self._viewer.add_image(
                                    zarr_da[:, cam, ...],
                                    name=cam_name,
                                    rgb=True,
                                )
                                entry.layer_names.append(cam_name)
                    else:
                        # (frames, H, W) or (frames, H, W, C)
                        vid_name = entry.layer_name("video")
                        if vid_name not in existing:
                            self._viewer.add_image(
                                zarr_da,
                                name=vid_name,
                                rgb=(zarr_da.ndim == 4),
                            )
                            entry.layer_names.append(vid_name)
                except Exception as _e:
                    QMessageBox.warning(self, "Zarr load error", str(_e))
            else:
                # ── Standard video (av / imageio) ────────────────────
                vid_name = entry.layer_name("video")
                if vid_name not in existing:
                    _vid_loaded = False
                    try:
                        import av           # type: ignore
                        import dask         # type: ignore
                        import dask.array as da  # type: ignore
                        import imageio.v3 as iio # type: ignore

                        # ── 1. Metadata only — no frame decoding yet ──────────
                        props = iio.improps(
                            str(entry.video_path), plugin="pyav"
                        )
                        n_frames = props.n_images
                        if not n_frames:
                            raise IOError("Could not read video metadata")
                        # props.shape may be (H, W, C) or (N, H, W, C);
                        # index from the end to get H and W reliably.
                        H, W = int(props.shape[-3]), int(props.shape[-2])
                        frame_dtype = props.dtype

                        # Detect grayscale by peeking at the first frame only
                        first = iio.imread(
                            str(entry.video_path), index=0, plugin="pyav"
                        )
                        grey = (
                            first.ndim == 3 and first.shape[2] == 3
                            and np.array_equal(first[:, :, 0], first[:, :, 1])
                        )
                        frame_shape = (H, W) if grey else (H, W, 3)

                        # ── 2. Chunk reader — seek once, read sequentially ────
                        # Each dask chunk opens the file, seeks to the nearest
                        # keyframe, then decodes forward.  Only the requested
                        # chunk is ever in RAM.
                        _CHUNK = 64  # frames per chunk

                        def _read_chunk(
                            path: str, start: int, count: int,
                            h: int, w: int, grey: bool, dtype,
                        ) -> np.ndarray:
                            import av as _av
                            import numpy as _np
                            out = _np.empty(
                                (count, h, w) if grey else (count, h, w, 3),
                                dtype=dtype,
                            )
                            written = 0
                            with _av.open(path) as cont:
                                stream = cont.streams.video[0]
                                fps = float(
                                    stream.average_rate
                                    or stream.guessed_rate
                                    or 25.0
                                )
                                tb = float(stream.time_base)
                                # seek backward to keyframe before start
                                if start > 0:
                                    target = int(start / fps / tb)
                                    cont.seek(
                                        target, stream=stream,
                                        backward=True, any_frame=False,
                                    )
                                for frame in cont.decode(stream):
                                    # Estimate frame index from pts
                                    fi = int(
                                        round(float(frame.pts or 0) * tb * fps)
                                    ) if frame.pts is not None else (start + written)
                                    if fi < start:
                                        continue
                                    if written >= count:
                                        break
                                    img = frame.to_ndarray(format="rgb24")
                                    out[written] = img[:, :, 0] if grey else img
                                    written += 1
                            # pad with zeros if video ended early
                            return out

                        # ── 3. Assemble lazy dask array ───────────────────────
                        _path = str(entry.video_path)
                        chunks = []
                        for start in range(0, n_frames, _CHUNK):
                            count = min(_CHUNK, n_frames - start)
                            cshape = (
                                (count, H, W) if grey else (count, H, W, 3)
                            )
                            chunks.append(
                                da.from_delayed(
                                    dask.delayed(_read_chunk)(
                                        _path, start, count,
                                        H, W, grey, frame_dtype,
                                    ),
                                    shape=cshape,
                                    dtype=frame_dtype,
                                )
                            )
                        vid_arr = da.concatenate(chunks, axis=0)
                        is_rgb = vid_arr.ndim == 4
                        self._viewer.add_image(vid_arr, name=vid_name, rgb=is_rgb)
                        _vid_loaded = True
                    except Exception as e:
                        QMessageBox.warning(self, "Video load error", str(e))
                    if _vid_loaded and vid_name not in entry.layer_names:
                        entry.layer_names.append(vid_name)

        # ── Pose file ────────────────────────────────────────────────
        if entry.pose_path:
            pts_name = entry.layer_name("points")
            # Parse the file if coords_data is not yet populated so other panels
            # (Analysis) can access it even when the Points layer already exists.
            _points_result = None
            if not entry.coords_data or pts_name not in existing:
                try:
                    _points_result = _load_pose_as_points(entry.pose_path)
                    entry.coords_data = _points_result[2]
                except Exception as e:
                    QMessageBox.warning(self, "Pose load error", str(e))
            if pts_name not in existing and _points_result is not None:
                try:
                    points, props, _ = _points_result
                    # size: ~1 % of frame height if a video layer exists, else 3 px
                    size = 3
                    for lyr in self._viewer.layers:
                        if lyr.name == entry.layer_name("video"):
                            try:
                                size = lyr.data.shape[-2] / 100
                            except Exception:
                                pass
                            break
                    self._viewer.add_points(
                        points, properties=props,
                        name=pts_name, size=size, ndim=3,
                        out_of_slice_display=False,
                    )
                    if pts_name not in entry.layer_names:
                        entry.layer_names.append(pts_name)
                except Exception as e:
                    QMessageBox.warning(self, "Pose load error", str(e))

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
                text += f"  [{Path(entry.video_path).name}]"
            item = QListWidgetItem(text)
            if entry.status == "active":
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._session_list.addItem(item)
