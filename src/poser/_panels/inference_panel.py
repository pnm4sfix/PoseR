"""
_panels/inference_panel.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Inference Panel — Pose Estimation and Behaviour Decoding.

Mode 1 — Pose Estimation:
  * Dropdown of pretrained YOLO11 and PoseR species models
    (zeb.pt / fly3.pt / mouse7.pt / mouse13.pt downloaded from GitHub releases)
  * Custom .pt browse
  * Number of individuals spinbox
  * Runs YOLO .track() on the active session video → adds Points layer

Mode 2 — Behaviour Decoding:
  * Browse for a .ckpt decoder checkpoint
  * Run inference on bouts detected by the Analysis panel
  * Batch mode across all session files
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

import napari
import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from poser.core.session import SessionManager
import torch

# ---------------------------------------------------------------------------
# Pretrained model registry
# ---------------------------------------------------------------------------

GITHUB_RELEASE_URL = "https://github.com/pnm4sfix/PoseR/releases/download/v0.0.1b4/"

POSE_MODELS = [
    "zeb.pt",
    "fly3.pt",
    "mouse7.pt",
    "mouse13.pt",
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
]

POSER_PRETRAINED = {"zeb.pt", "fly3.pt", "mouse7.pt", "mouse13.pt"}


def _resolve_model(name: str):
    """Return a YOLO model instance, downloading PoseR releases as needed."""
    from ultralytics import YOLO  # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name in POSER_PRETRAINED:
        url = GITHUB_RELEASE_URL + name
        return YOLO(url)
    return YOLO(name).to(device)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class InferencePanel(QWidget):
    """Napari dock widget for Pose Estimation and Behaviour Decoding."""

    _progress_signal = Signal(int)
    _status_signal = Signal(str)
    _imgsz_signal = Signal(int)

    def __init__(
        self,
        viewer: napari.Viewer,
        session: Optional[SessionManager] = None,
        bouts_getter=None,
    ):
        super().__init__()
        self._viewer = viewer
        self._session = session or SessionManager(viewer=viewer)
        self._bouts_getter = bouts_getter
        self._behaviour_checkpoint: Optional[Path] = None
        self._predictions: Dict[int, int] = {}
        self._add_points_signal_data = None

        self._build_ui()
        self._progress_signal.connect(self._on_progress)
        self._status_signal.connect(self._on_status)
        self._imgsz_signal.connect(self._on_imgsz_detected)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # ── Mode selector ─────────────────────────────────────────────
        mode_grp = QGroupBox("Mode")
        mode_lay = QHBoxLayout(mode_grp)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Pose Estimation", "Behaviour Decoding"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_lay.addWidget(self._mode_combo)
        root.addWidget(mode_grp)

        # ── Stacked pages ─────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_pose_page())
        self._stack.addWidget(self._build_behaviour_page())
        root.addWidget(self._stack)

        root.addStretch()

    def _build_pose_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        model_grp = QGroupBox("Pose Model")
        model_lay = QVBoxLayout(model_grp)
        model_lay.addWidget(QLabel("Pretrained model:"))
        self._pose_model_combo = QComboBox()
        self._pose_model_combo.addItems(POSE_MODELS)
        self._pose_model_combo.currentTextChanged.connect(self._on_pose_model_selected)
        model_lay.addWidget(self._pose_model_combo)
        self._pose_model_label = QLabel(
            "zeb / fly3 / mouse7 / mouse13 are PoseR species models "
            "downloaded automatically from GitHub on first use."
        )
        self._pose_model_label.setStyleSheet("font-size: 10px; color: grey;")
        self._pose_model_label.setWordWrap(True)
        model_lay.addWidget(self._pose_model_label)
        btn_browse_pose = QPushButton("Browse for custom .pt…")
        btn_browse_pose.clicked.connect(self._on_browse_pose_model)
        model_lay.addWidget(btn_browse_pose)
        lay.addWidget(model_grp)

        opts_grp = QGroupBox("Options")
        opts_lay = QVBoxLayout(opts_grp)

        inds_row = QHBoxLayout()
        inds_row.addWidget(QLabel("Max individuals:"))
        self._n_inds_spin = QSpinBox()
        self._n_inds_spin.setRange(1, 100)
        self._n_inds_spin.setValue(1)
        inds_row.addWidget(self._n_inds_spin)
        inds_row.addStretch()
        opts_lay.addLayout(inds_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Inference mode:"))
        self._infer_mode_combo = QComboBox()
        self._infer_mode_combo.addItems(["predict  (fast / no tracking)",
                                         "track  (sequential / with tracking)"])
        self._infer_mode_combo.currentIndexChanged.connect(self._on_infer_mode_changed)
        mode_row.addWidget(self._infer_mode_combo)
        opts_lay.addLayout(mode_row)

        self._infer_mode_desc = QLabel(
            "predict: frames batched in parallel — fast but individual IDs "
            "are not linked across frames.\n"
            "track: frame-by-frame with ByteTrack — slower but each animal "
            "keeps a consistent ID throughout the video."
        )
        self._infer_mode_desc.setStyleSheet("font-size: 10px; color: grey;")
        self._infer_mode_desc.setWordWrap(True)
        opts_lay.addWidget(self._infer_mode_desc)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch size (predict only):"))
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 256)
        self._batch_spin.setValue(16)
        batch_row.addWidget(self._batch_spin)
        batch_row.addStretch()
        opts_lay.addLayout(batch_row)

        imgsz_row = QHBoxLayout()
        imgsz_row.addWidget(QLabel("Image size (imgsz):"))
        self._imgsz_spin = QSpinBox()
        self._imgsz_spin.setRange(0, 8192)
        self._imgsz_spin.setValue(0)
        self._imgsz_spin.setSingleStep(32)
        self._imgsz_spin.setToolTip(
            "Inference image size passed to YOLO.\n"
            "0 = auto-detect from the loaded model's training metadata.\n"
            "Populated automatically after the model is loaded; override here if needed."
        )
        imgsz_row.addWidget(self._imgsz_spin)
        self._imgsz_auto_label = QLabel("(auto)")
        self._imgsz_auto_label.setStyleSheet("font-size: 10px; color: grey;")
        imgsz_row.addWidget(self._imgsz_auto_label)
        imgsz_row.addStretch()
        opts_lay.addLayout(imgsz_row)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Zarr camera axis:"))
        self._zarr_cam_spin = QSpinBox()
        self._zarr_cam_spin.setRange(0, 31)
        self._zarr_cam_spin.setValue(0)
        self._zarr_cam_spin.setToolTip(
            "Which camera (axis-1 index) to run inference on when the\n"
            "active video is a zarr array store with shape\n"
            "(frames, cameras, H, W, C).  Ignored for normal video files."
        )
        cam_row.addWidget(self._zarr_cam_spin)
        cam_row.addStretch()
        opts_lay.addLayout(cam_row)

        lay.addWidget(opts_grp)

        btn_run_pose = QPushButton("▶  Run Pose Estimation (active video)")
        btn_run_pose.setStyleSheet("font-weight: bold;")
        btn_run_pose.clicked.connect(self._on_run_pose)
        lay.addWidget(btn_run_pose)

        btn_run_pose_batch = QPushButton("▶▶  Run Pose Estimation (all session videos)")
        btn_run_pose_batch.clicked.connect(self._on_run_pose_batch)
        lay.addWidget(btn_run_pose_batch)

        self._pose_progress = QProgressBar()
        self._pose_progress.setRange(0, 100)
        self._pose_progress.setValue(0)
        self._pose_progress.setVisible(False)
        lay.addWidget(self._pose_progress)

        self._pose_log = QTextEdit()
        self._pose_log.setReadOnly(True)
        self._pose_log.setFixedHeight(100)
        self._pose_log.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; font-family: monospace; }"
        )
        lay.addWidget(self._pose_log)

        return page

    def _build_behaviour_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        ckpt_grp = QGroupBox("Decoder Checkpoint (.ckpt)")
        ckpt_lay = QVBoxLayout(ckpt_grp)
        self._ckpt_label = QLabel("No checkpoint loaded.")
        self._ckpt_label.setStyleSheet("font-size: 10px; color: grey;")
        self._ckpt_label.setWordWrap(True)
        ckpt_lay.addWidget(self._ckpt_label)
        btn_browse = QPushButton("Browse for checkpoint (.ckpt)…")
        btn_browse.clicked.connect(self._on_browse_checkpoint)
        ckpt_lay.addWidget(btn_browse)
        lay.addWidget(ckpt_grp)

        single_grp = QGroupBox("Single File Inference")
        single_lay = QVBoxLayout(single_grp)
        btn_predict = QPushButton("▶  Predict Behaviours (active file)")
        btn_predict.clicked.connect(self._on_predict_single)
        single_lay.addWidget(btn_predict)
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        single_lay.addWidget(self._result_label)
        lay.addWidget(single_grp)

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
        self._status_text.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #d4d4d4; font-family: monospace; }"
        )
        batch_lay.addWidget(self._status_text)
        lay.addWidget(batch_grp)

        export_grp = QGroupBox("Export Predictions")
        export_lay = QHBoxLayout(export_grp)
        btn_export = QPushButton("Save Predictions (JSON)…")
        btn_export.clicked.connect(self._on_export_predictions)
        export_lay.addWidget(btn_export)
        lay.addWidget(export_grp)

        return page

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Zarr frame generator
    # ------------------------------------------------------------------

    @staticmethod
    def _zarr_frame_generator(zarr_array, cam: int):
        """Yield individual frames from a zarr array for YOLO.

        Supports two layouts:
          - 5-D: (frames, cams, H, W, C)  — yields zarr_array[i, cam, ...]
          - 4-D: (frames, H, W, C)         — yields zarr_array[i, ...]
          - 3-D: (frames, H, W)            — yields zarr_array[i, ...]
        Each yielded frame is a contiguous uint8 numpy array.
        """
        ndim = zarr_array.ndim
        n_frames = zarr_array.shape[0]
        for i in range(n_frames):
            if ndim == 5:
                frame = np.ascontiguousarray(zarr_array[i, cam], dtype=np.uint8)
            else:
                frame = np.ascontiguousarray(zarr_array[i], dtype=np.uint8)
            yield frame

    # ------------------------------------------------------------------
    # Pose Estimation slots
    # ------------------------------------------------------------------

    def _on_infer_mode_changed(self, idx: int) -> None:
        self._batch_spin.setEnabled(idx == 0)  # only relevant for predict

    def _on_imgsz_detected(self, imgsz: int) -> None:
        """Called on the main thread when a model's training imgsz is read."""
        # Only auto-populate if the user hasn't set a manual override (i.e. still 0)
        if self._imgsz_spin.value() == 0:
            self._imgsz_spin.setValue(imgsz)
        self._imgsz_auto_label.setText(f"(model default: {imgsz})")

    def _on_pose_model_selected(self, name: str) -> None:
        if name in POSER_PRETRAINED:
            self._pose_model_label.setText(
                f"Will download {name} from GitHub releases on first use."
            )
        else:
            self._pose_model_label.setText(f"Model: {name}")

    def _on_browse_pose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select pose model", "", "YOLO weights (*.pt);;All files (*)"
        )
        if path:
            self._pose_model_combo.insertItem(0, path)
            self._pose_model_combo.setCurrentIndex(0)

    def _on_run_pose(self) -> None:
        try:
            self._run_pose_impl()
        except Exception as exc:
            import traceback
            msg = f"ERROR: {exc}\n{traceback.format_exc()}"
            self._pose_log.setVisible(True)
            self._pose_log.append(msg)
            QMessageBox.critical(self, "Pose estimation error", msg)

    def _run_pose_impl(self) -> None:
        # Always make log visible first so any message is seen
        self._pose_progress.setVisible(True)
        self._pose_progress.setValue(0)
        self._pose_log.clear()
        msg0 = "▶ Button clicked — checking session …"
        self._pose_log.append(msg0)

        entry = self._session.active
        if entry is None:
            msg = (
                "✖ No active session entry.\n"
                "  → Go to the Data Panel, add a video, and double-click the row to activate it."
            )
            self._pose_log.append(msg)
            return
        video_path = entry.video_path
        if not video_path:
            msg = (
                "✖ Active session entry has no video path.\n"
                "  → Use Data Panel → Add Video(s)… or Add Zarr Video… first."
            )
            self._pose_log.append(msg)
            return
        model_name = self._pose_model_combo.currentText()
        n_inds = self._n_inds_spin.value()
        use_track = self._infer_mode_combo.currentIndex() == 1
        batch_size = self._batch_spin.value()
        zarr_cam = self._zarr_cam_spin.value()
        self._pose_log.append(
            f"▶ Starting: {Path(video_path).name}  |  model={model_name}  |  "
            f"n_inds={n_inds}  |  {'track' if use_track else 'predict'}"
        )

        def run():
            try:
                import torch
                self._status_signal.emit(f"Loading model: {model_name} …")

                model = _resolve_model(model_name)

                # --- Resolve imgsz: model metadata first, then spinbox override ---
                model_imgsz = 640
                try:
                    ov = model.overrides
                    raw = ov.get("imgsz", 640)
                    model_imgsz = raw[0] if isinstance(raw, (list, tuple)) else int(raw)
                except Exception:
                    pass
                user_imgsz = self._imgsz_spin.value()
                imgsz = user_imgsz if user_imgsz > 0 else model_imgsz
                self._imgsz_signal.emit(model_imgsz)  # update spinbox on main thread

                is_zarr = str(video_path).endswith(".zarr")

                if is_zarr:
                    import zarr  # type: ignore
                    _zarr_arr = zarr.open_array(str(video_path), mode="r")
                    total_frames = int(_zarr_arr.shape[0])
                    cam_label = f"cam{zarr_cam}" if _zarr_arr.ndim == 5 else ""
                    self._status_signal.emit(
                        f"Running on {Path(video_path).name}{f' [{cam_label}]' if cam_label else ''}"
                        f"  ({total_frames} frames, imgsz={imgsz}) …"
                    )
                    stem = f"{Path(video_path).stem}{'_' + cam_label if cam_label else ''}"

                    # Determine zarr chunk size along frame axis so we
                    # decompress each chunk exactly once instead of once per batch.
                    _chunk_frames = int(_zarr_arr.chunks[0]) if hasattr(_zarr_arr, "chunks") else 256

                    def _get_chunk(chunk_start, chunk_end):
                        """Read a full zarr chunk in one I/O call → (N, H, W, C) uint8."""
                        if _zarr_arr.ndim == 5:
                            return np.ascontiguousarray(
                                _zarr_arr[chunk_start:chunk_end, zarr_cam], dtype=np.uint8
                            )
                        return np.ascontiguousarray(
                            _zarr_arr[chunk_start:chunk_end], dtype=np.uint8
                        )

                    # Build a unified (frame_idx, result) iterator for zarr
                    def _zarr_results():
                        if use_track:
                            # For track: read chunk-aligned blocks, feed YOLO one frame at a time
                            for chunk_start in range(0, total_frames, _chunk_frames):
                                chunk_end = min(chunk_start + _chunk_frames, total_frames)
                                chunk_data = _get_chunk(chunk_start, chunk_end)
                                for local_i in range(len(chunk_data)):
                                    fi = chunk_start + local_i
                                    res = model.track(
                                        source=chunk_data[local_i],
                                        imgsz=imgsz,
                                        max_det=n_inds,
                                        persist=True,
                                        verbose=False,
                                    )
                                    yield fi, res[0]
                        else:
                            # For predict: read chunk-aligned blocks, feed YOLO in sub-batches
                            for chunk_start in range(0, total_frames, _chunk_frames):
                                chunk_end = min(chunk_start + _chunk_frames, total_frames)
                                chunk_data = _get_chunk(chunk_start, chunk_end)
                                n_in_chunk = len(chunk_data)
                                for sub in range(0, n_in_chunk, batch_size):
                                    sub_end = min(sub + batch_size, n_in_chunk)
                                    frames = [chunk_data[j] for j in range(sub, sub_end)]
                                    batch_res = model.predict(
                                        source=frames,
                                        imgsz=imgsz,
                                        max_det=n_inds,
                                        verbose=False,
                                    )
                                    for fi_off, res in enumerate(batch_res):
                                        yield chunk_start + sub + fi_off, res

                    results_iter = _zarr_results()

                else:
                    import cv2
                    cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    cap.release()
                    self._status_signal.emit(
                        f"Running on {Path(video_path).name} (imgsz={imgsz}) …"
                    )
                    stem = Path(video_path).stem

                    if use_track:
                        _raw = model.track(
                            source=str(video_path),
                            imgsz=imgsz,
                            stream=True,
                            max_det=n_inds,
                        )
                    else:
                        _raw = model.predict(
                            source=str(video_path),
                            imgsz=imgsz,
                            stream=True,
                            batch=batch_size,
                            max_det=n_inds,
                        )
                    results_iter = enumerate(_raw)

                points_list, conf_list, ind_list, node_list = [], [], [], []
                for frame_idx, result in results_iter:
                    if frame_idx % max(1, total_frames // 20) == 0:
                        self._progress_signal.emit(
                            int(100 * frame_idx / total_frames)
                        )
                    kp = result.keypoints
                    if kp is None:
                        continue
                    xy = kp.xy; device = xy.device
                    conf = (kp.conf if kp.conf is not None
                            else torch.ones(xy.shape[:2], device=device))
                    if use_track:
                        track_ids = (
                            result.boxes.id.to(device).int()
                            if result.boxes is not None and result.boxes.id is not None
                            else torch.arange(xy.shape[0], device=device)
                        )
                    else:
                        track_ids = torch.arange(xy.shape[0], device=device)
                    P, K, _ = xy.shape
                    xy_flat = xy.reshape(-1, 2)
                    pts = torch.stack(
                        (torch.full((P * K,), frame_idx, device=device),
                         xy_flat[:, 1], xy_flat[:, 0]), dim=1
                    )
                    points_list.append(pts)
                    conf_list.append(conf.reshape(-1))
                    ind_list.append(torch.repeat_interleave(track_ids, K))
                    node_list.append(torch.arange(K, device=device).repeat(P))

                if points_list:
                    self._add_points_signal_data = (
                        torch.cat(points_list).cpu().numpy(),
                        {
                            "confidence": torch.cat(conf_list).cpu().numpy(),
                            "ind": torch.cat(ind_list).cpu().numpy(),
                            "node": torch.cat(node_list).cpu().numpy(),
                        },
                        f"{stem}_pose",
                        entry,
                    )
                    self._status_signal.emit("__ADD_POINTS__")
                    self._status_signal.emit(
                        f"Done — {len(points_list)} frames with keypoints."
                    )
                else:
                    self._status_signal.emit("No keypoints detected.")
                self._progress_signal.emit(100)

            except Exception as exc:
                import traceback
                self._status_signal.emit(f"ERROR: {exc}\n{traceback.format_exc()}")

        threading.Thread(target=run, daemon=True).start()

    def _on_run_pose_batch(self) -> None:
        entries = [e for e in self._session.entries if e.video_path]
        if not entries:
            QMessageBox.warning(self, "No videos",
                                "No session entries have a video file.")
            return
        model_name = self._pose_model_combo.currentText()
        n_inds = self._n_inds_spin.value()
        use_track = self._infer_mode_combo.currentIndex() == 1
        batch_size = self._batch_spin.value()
        zarr_cam = self._zarr_cam_spin.value()
        self._pose_progress.setVisible(True)
        self._pose_progress.setValue(0)
        self._pose_log.clear()

        def run():
            try:
                self._status_signal.emit(f"Loading model: {model_name} …")
                model = _resolve_model(model_name)

                # --- Resolve imgsz ---
                model_imgsz = 640
                try:
                    ov = model.overrides
                    raw = ov.get("imgsz", 640)
                    model_imgsz = raw[0] if isinstance(raw, (list, tuple)) else int(raw)
                except Exception:
                    pass
                user_imgsz = self._imgsz_spin.value()
                imgsz = user_imgsz if user_imgsz > 0 else model_imgsz
                self._imgsz_signal.emit(model_imgsz)

                import torch
                for i, entry in enumerate(entries):
                    is_zarr = str(entry.video_path).endswith(".zarr")

                    if is_zarr:
                        import zarr  # type: ignore
                        _zarr_arr = zarr.open_array(str(entry.video_path), mode="r")
                        _total = int(_zarr_arr.shape[0])
                        cam_label = f"cam{zarr_cam}" if _zarr_arr.ndim == 5 else ""
                        self._status_signal.emit(
                            f"[{i+1}/{len(entries)}] {Path(entry.video_path).name}"
                            f"{f' [{cam_label}]' if cam_label else ''}"
                        )
                        stem = f"{Path(entry.video_path).stem}{'_' + cam_label if cam_label else ''}"
                        _zc = zarr_cam
                        _cf = int(_zarr_arr.chunks[0]) if hasattr(_zarr_arr, "chunks") else 256

                        def _get_zchunk(cs, ce, _arr=_zarr_arr, _cam=_zc):
                            if _arr.ndim == 5:
                                return np.ascontiguousarray(_arr[cs:ce, _cam], dtype=np.uint8)
                            return np.ascontiguousarray(_arr[cs:ce], dtype=np.uint8)

                        def _zarr_batch_results(_tot=_total, _chunk_f=_cf):
                            if use_track:
                                for cs in range(0, _tot, _chunk_f):
                                    ce = min(cs + _chunk_f, _tot)
                                    chunk = _get_zchunk(cs, ce)
                                    for local_i in range(len(chunk)):
                                        res = model.track(
                                            source=chunk[local_i],
                                            imgsz=imgsz, max_det=n_inds,
                                            persist=True, verbose=False,
                                        )
                                        yield cs + local_i, res[0]
                            else:
                                for cs in range(0, _tot, _chunk_f):
                                    ce = min(cs + _chunk_f, _tot)
                                    chunk = _get_zchunk(cs, ce)
                                    for bs in range(0, len(chunk), batch_size):
                                        be = min(bs + batch_size, len(chunk))
                                        frames = [chunk[j] for j in range(bs, be)]
                                        for fi_off, res in enumerate(
                                            model.predict(source=frames, imgsz=imgsz,
                                                          max_det=n_inds, verbose=False)
                                        ):
                                            yield cs + bs + fi_off, res

                        results_iter = _zarr_batch_results()
                    else:
                        import cv2
                        self._status_signal.emit(
                            f"[{i+1}/{len(entries)}] {Path(entry.video_path).name}"
                        )
                        cap = cv2.VideoCapture(str(entry.video_path))
                        _total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                        cap.release()
                        stem = Path(entry.video_path).stem

                        if use_track:
                            _raw = model.track(
                                source=str(entry.video_path),
                                imgsz=imgsz, stream=True, max_det=n_inds,
                            )
                        else:
                            _raw = model.predict(
                                source=str(entry.video_path),
                                imgsz=imgsz, stream=True,
                                batch=batch_size, max_det=n_inds,
                            )
                        results_iter = enumerate(_raw)

                    points_list, conf_list, ind_list, node_list = [], [], [], []
                    for frame_idx, result in results_iter:
                        kp = result.keypoints
                        if kp is None:
                            continue
                        xy = kp.xy; device = xy.device
                        conf = (kp.conf if kp.conf is not None
                                else torch.ones(xy.shape[:2], device=device))
                        if use_track:
                            track_ids = (
                                result.boxes.id.to(device).int()
                                if result.boxes is not None and result.boxes.id is not None
                                else torch.arange(xy.shape[0], device=device)
                            )
                        else:
                            track_ids = torch.arange(xy.shape[0], device=device)
                        P, K, _ = xy.shape
                        xy_flat = xy.reshape(-1, 2)
                        pts = torch.stack(
                            (torch.full((P*K,), frame_idx, device=device),
                             xy_flat[:,1], xy_flat[:,0]), dim=1
                        )
                        points_list.append(pts)
                        conf_list.append(conf.reshape(-1))
                        ind_list.append(torch.repeat_interleave(track_ids, K))
                        node_list.append(torch.arange(K, device=device).repeat(P))

                    if points_list:
                        self._add_points_signal_data = (
                            torch.cat(points_list).cpu().numpy(),
                            {
                                "confidence": torch.cat(conf_list).cpu().numpy(),
                                "ind": torch.cat(ind_list).cpu().numpy(),
                                "node": torch.cat(node_list).cpu().numpy(),
                            },
                            f"{stem}_pose",
                            entry,
                        )
                        self._status_signal.emit("__ADD_POINTS__")

                    self._progress_signal.emit(int(100 * (i + 1) / len(entries)))

                self._status_signal.emit("Batch pose estimation complete.")
            except Exception as exc:
                import traceback
                self._status_signal.emit(f"ERROR: {exc}\n{traceback.format_exc()}")

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # Behaviour Decoding slots
    # ------------------------------------------------------------------

    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "", "Checkpoints (*.ckpt *.pt);;All files (*)"
        )
        if path:
            self._behaviour_checkpoint = Path(path)
            self._ckpt_label.setText(str(self._behaviour_checkpoint))

    def _on_predict_single(self) -> None:
        if not self._behaviour_checkpoint:
            QMessageBox.warning(self, "No checkpoint",
                                "Please select a decoder checkpoint first.")
            return
        entry = self._session.active
        if entry is None or not entry.coords_data:
            QMessageBox.warning(self, "No data",
                                "No pose data loaded in the active session entry.")
            return
        bouts = self._bouts_getter() if self._bouts_getter else []
        if not bouts:
            QMessageBox.warning(self, "No bouts",
                                "Run bout detection in the Analysis Panel first.")
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
        if not self._behaviour_checkpoint:
            QMessageBox.warning(self, "No checkpoint",
                                "Please select a decoder checkpoint first.")
            return
        if not self._session.entries:
            QMessageBox.warning(self, "Empty session",
                                "Add files to the session first.")
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
                checkpoint=self._behaviour_checkpoint,
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
                json.dump(
                    {str(k): int(v) for k, v in self._predictions.items()},
                    fh, indent=2,
                )

    # ------------------------------------------------------------------
    # Signal receivers (always called on the main Qt thread)
    # ------------------------------------------------------------------

    def _on_progress(self, value: int) -> None:
        if self._mode_combo.currentIndex() == 0:
            self._pose_progress.setValue(value)
        else:
            self._progress_bar.setValue(value)

    def _on_status(self, text: str) -> None:
        if text == "__ADD_POINTS__":
            self._add_points_layer()
            return
        if self._mode_combo.currentIndex() == 0:
            self._pose_log.append(text)
        else:
            self._status_text.append(text)

    def _add_points_layer(self) -> None:
        """Add / replace a points layer on the main Qt thread, then save pose file."""
        if self._add_points_signal_data is None:
            return
        points, props, layer_name, entry = self._add_points_signal_data
        for lyr in list(self._viewer.layers):
            if lyr.name == layer_name:
                self._viewer.layers.remove(lyr)
        self._viewer.add_points(
            points,
            properties=props,
            name=layer_name,
            size=3,
            opacity=0.8,
            out_of_slice_display=False,
        )
        # Save PoseR pose file and link it to the session entry
        if entry is not None and entry.video_path:
            try:
                save_path, coords_data = self._save_poser_pose_file(
                    points, props, entry
                )
                entry.pose_path = str(save_path)
                entry.coords_data = coords_data
                self._pose_log.append(
                    f"Saved pose file: {save_path.name}"
                )
            except Exception as exc:
                self._pose_log.append(f"Warning: could not save pose file — {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_poser_pose_file(
        self, points: np.ndarray, props: dict, entry
    ):
        """Write a PoseR-format PyTables HDF5 file next to the video.

        Layout: one root group per individual (ind1, ind2 …), each containing
        a ``coords`` array of shape (3, n_nodes, n_frames) where axis-0 is
        [x, y, confidence].

        Returns
        -------
        save_path : Path
        coords_data : dict matching the format consumed by _coords_data_to_points
        """
        import pandas as pd
        import tables as tb  # type: ignore  (PyTables)

        video_path = Path(entry.video_path)
        save_path = video_path.parent / f"{video_path.stem}_pose.h5"

        frames = points[:, 0].astype(int)
        ys = points[:, 1].astype(np.float32)
        xs = points[:, 2].astype(np.float32)
        confs = props["confidence"].astype(np.float32)
        inds = props["ind"].astype(int)
        nodes = props["node"].astype(int)

        n_frames = int(frames.max()) + 1
        n_nodes = int(nodes.max()) + 1
        unique_inds = np.unique(inds)

        coords_data = {}
        filters = tb.Filters(complevel=5, complib="blosc")
        with tb.open_file(str(save_path), mode="w", filters=filters) as f:
            for ind_id in unique_inds:
                mask = inds == ind_id
                f_m = frames[mask]
                x_m = xs[mask]
                y_m = ys[mask]
                c_m = confs[mask]
                n_m = nodes[mask]

                arr = np.full((3, n_nodes, n_frames), np.nan, dtype=np.float32)
                arr[0, n_m, f_m] = x_m   # x
                arr[1, n_m, f_m] = y_m   # y
                arr[2, n_m, f_m] = c_m   # confidence

                ind_name = f"ind{int(ind_id) + 1}"
                grp = f.create_group("/", ind_name)
                f.create_array(grp, "coords", arr)

                coords_data[ind_name] = {
                    "x": pd.DataFrame(arr[0]),
                    "y": pd.DataFrame(arr[1]),
                    "ci": pd.DataFrame(arr[2]),
                }

        return save_path, coords_data

    def _predict_from_array(self, X: np.ndarray, bouts: List[dict]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from poser.models.registry import load_model
        from poser.api import PoseR

        poser = PoseR()
        mcfg = poser.config.model
        model = load_model(
            mcfg.architecture,
            checkpoint=self._behaviour_checkpoint,
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

