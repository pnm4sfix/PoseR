"""YOLO-pose estimation over a video, producing PoseR coords.

PoseEstimator runs it step by step, for the Inference panel and BatchJob: load a
model, open a video or zarr array, run YOLO over its frames, and gather the
keypoints into coords.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import zarr

from .schemas.pose_estimation import FrameKeypoints, InferenceMode

log = logging.getLogger(__name__)

GITHUB_RELEASE_URL = "https://github.com/pnm4sfix/PoseR/releases/download/v0.0.1b4/"
# Tuples, not sets: the Inference panel lists them in this order.
POSER_PRETRAINED = ("zeb.pt", "fly3.pt", "mouse7.pt", "mouse13.pt")
YOLO_PRETRAINED = (
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
)


class PoseEstimator:
    """Pose estimation over one video: load a model, open the input, read frames."""

    def __init__(self) -> None:
        self._model = None
        self._source = None  # video Path, or an open zarr array
        self._camera = 0
        self._n_frames = 0

    def model_load(self, path: Path, device: torch.device) -> None:
        """Load a YOLO pose model onto device.

        Args:
            path: A local .pt file, or the bare name of a pretrained model. The
                PoseR models download from the GitHub release and the
                yolo11*-pose models from ultralytics, each on first use.
            device: Where to run, normally core.settings.resolve_device().

        Raises:
            FileNotFoundError: If path is neither a file nor a pretrained name.
        """
        path = Path(path)
        if path.exists():
            source = str(path)
        elif str(path) in POSER_PRETRAINED:
            source = GITHUB_RELEASE_URL + str(path)
        elif str(path) in YOLO_PRETRAINED:
            source = str(path)
        else:
            raise FileNotFoundError(
                f"Pose model not found: {path}. Pick a .pt file or one of "
                f"{[*POSER_PRETRAINED, *YOLO_PRETRAINED]}."
            )

        # lazy: test_batch.py pins that the decode path never imports ultralytics
        from ultralytics import YOLO

        self._model = YOLO(source).to(device)

    def input_load(self, path: str | Path, camera: int = 0) -> int:
        """Open a video file or zarr array without reading its frames.

        Frames are read later, one at a time, by iter_frames, so memory use
        does not grow with video length.

        Args:
            path: A video file, or a zarr array store shaped (frames, H, W),
                (frames, H, W, C) or (frames, cameras, H, W, C).
            camera: Which camera of a 5-D zarr array to use. Ignored otherwise.

        Returns:
            The number of frames, for progress. For a video file it is OpenCV's
            count, which some formats only estimate.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If the video cannot be opened, the zarr array has an
                unsupported shape, or camera is out of range.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        # TODO: not sure whether anybody is using zarr filetype for pose estimation download, so i will need to ask Pierce about that, probably that funcitonality are redundant
        if path.suffix == ".zarr":
            source = zarr.open_array(str(path), mode="r")
            if source.ndim not in (3, 4, 5):
                raise ValueError(
                    f"{path.name} has shape {source.shape}. Expected (frames, H, W), "
                    "(frames, H, W, C) or (frames, cameras, H, W, C)."
                )
            if source.ndim == 5 and not 0 <= camera < source.shape[1]:
                raise ValueError(
                    f"Camera {camera} requested, but {path.name} has "
                    f"{source.shape[1]} cameras (0 to {source.shape[1] - 1})."
                )
            n_frames = int(source.shape[0])
        else:
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    raise ValueError(f"Could not open video: {path}")
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                cap.release()
            source = path

        self._source = source
        self._camera = camera
        self._n_frames = n_frames
        return n_frames

    # TODO: probably would be a good idea made a base iterator for this thing but not for now
    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield the opened input's frames in order, one uint8 array each.

        A video yields (H, W, 3) BGR frames as OpenCV decodes them. A zarr
        array yields (H, W) or (H, W, C) frames, from the chosen camera when it
        has a camera axis. It is read one chunk at a time, so each chunk is
        decompressed once rather than once per frame.

        Raises:
            RuntimeError: If input_load has not been called.
        """
        if self._source is None:
            raise RuntimeError("No input opened. Call input_load first.")

        if isinstance(self._source, Path):
            cap = cv2.VideoCapture(str(self._source))
            try:
                ok, frame = cap.read()
                while ok:
                    yield frame
                    ok, frame = cap.read()
            finally:
                cap.release()
            return

        chunk_len = self._source.chunks[0]
        for start in range(0, self._n_frames, chunk_len):
            stop = min(start + chunk_len, self._n_frames)
            if self._source.ndim == 5:
                block = self._source[start:stop, self._camera]
            else:
                block = self._source[start:stop]
            yield from np.ascontiguousarray(block, dtype=np.uint8)

    def iter_keypoints(
        self,
        mode: InferenceMode = InferenceMode.PREDICT,
        batch_size: int = 16,
        max_individuals: int = 1,
        imgsz: int | None = None,
    ) -> Iterator[FrameKeypoints]:
        """Run YOLO over every frame and yield what it found, frame by frame.

        Args:
            mode: PREDICT treats every frame on its own, batch_size frames per
                call. TRACK runs one frame at a time and follows each animal
                across frames, so its ID stays the same.
            batch_size: Frames per call in predict mode. Track ignores it.
            max_individuals: The most animals to report per frame, passed to
                YOLO as max_det.
            imgsz: Image size, a multiple of 32. None uses the model's own
                training size.

        Raises:
            RuntimeError: If model_load or input_load has not been called.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call model_load first.")

        options = {"max_det": max_individuals, "verbose": False}
        # imgsz=None would override the model's own training size.
        if imgsz is not None:
            options["imgsz"] = imgsz

        frames = self.iter_frames()
        if mode == InferenceMode.TRACK:
            for index, frame in enumerate(frames):
                # persist=False on a video's first frame starts a fresh tracker,
                # so IDs carry from frame to frame but never between videos.
                results = self._model.track(frame, persist=index > 0, **options)
                yield self._keypoints_from_result(results[0], index)
        
        if mode == InferenceMode.PREDICT:
            start = 0
            while batch := list(islice(frames, batch_size)):
                for offset, result in enumerate(self._model.predict(batch, **options)):
                    yield self._keypoints_from_result(result, start + offset)
                start += len(batch)

    @staticmethod
    def coords_from_keypoints(
        records: list[FrameKeypoints],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Gather per-frame keypoints into whole-video coords, one per animal.

        Individuals are named ind1, ind2, … in the order they first appear,
        since raw IDs count from 0 in predict mode and from 1 in track mode.
        Frames where an animal was not found stay NaN.

        Args:
            records: One per video frame, in frame order, as iter_keypoints
                yields them. Their count sets the length, so frames with no
                detection at the end of the video still count.

        Returns:
            {individual: {"x", "y", "ci"}}, each (V, T): the PoseR-native
            coords that save_coords_to_h5 writes and read_coords returns.
        """
        n_frames = len(records)
        n_nodes = max((record.xy.shape[1] for record in records), default=0)
        coords: dict[str, dict[str, np.ndarray]] = {}
        names: dict[int, str] = {}  # raw ID -> ind1, ind2, …
        for record in records:
            for p, raw_id in enumerate(record.ids.tolist()):
                if raw_id not in names:
                    names[raw_id] = f"ind{len(names) + 1}"
                    coords[names[raw_id]] = {
                        key: np.full((n_nodes, n_frames), np.nan, dtype=np.float32)
                        for key in ("x", "y", "ci")
                    }
                animal = coords[names[raw_id]]
                animal["x"][:, record.frame] = record.xy[p, :, 0]
                animal["y"][:, record.frame] = record.xy[p, :, 1]
                animal["ci"][:, record.frame] = record.conf[p]
        return coords

    @staticmethod
    def points_from_coords(
        coords_data: dict,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Convert coords_data to the data and properties of a napari Points layer.

        Mirrors widget.get_points(): z = frame index tiled across body-part rows.

        Args:
            coords_data: {individual: {"x", "y", "ci"}}, each (V, T), as numpy
                arrays or DataFrames.

        Returns:
            The (N, 3) points as frame, y, x, and their properties: confidence,
            ind (the individual's position in coords_data, from 0) and node.
        """
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

    @staticmethod
    def _keypoints_from_result(result, frame: int) -> FrameKeypoints:
        """Convert one frame's YOLO result into numpy keypoints.

        Args:
            result: One ultralytics Results object, covering a single frame.
            frame: Index of that frame in the video. YOLO cannot know it,
                because it is handed bare arrays.
        """
        # Using .cpu() for avoiding crashing program on gpu
        result = result.cpu().numpy()
        kp = result.keypoints
        if kp is None:
            return FrameKeypoints(
                frame=frame,
                xy=np.empty((0, 0, 2), dtype=np.float32),
                conf=np.empty((0, 0), dtype=np.float32),
                ids=np.empty(0, dtype=np.int64),
            )
        xy = kp.xy.astype(np.float32)
        # A model trained without per-point visibility gives no confidence.
        if kp.conf is not None:
            conf = kp.conf.astype(np.float32)
        else:
            conf = np.ones(xy.shape[:2], dtype=np.float32)
        # Only track mode assigns IDs, and not on every frame.
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.astype(np.int64)
        else:
            ids = np.arange(xy.shape[0], dtype=np.int64)
        return FrameKeypoints(frame=frame, xy=xy, conf=conf, ids=ids)
