"""YOLO-pose estimation over a video, producing a PoseR coords file.

Extracted verbatim from BatchJob._run_pose_estimation. REFACTOR_PLAN Phase 3.2
consolidates the _widget.py pose estimation into this module; the known defects
listed there are deliberately preserved here so the extraction stays a pure
move.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .io import save_coords_to_h5

log = logging.getLogger(__name__)


def estimate_poses_from_video(
    video_path: str,
    checkpoint: str = "",
    n_individuals: int = 1,
) -> str:
    """Track poses through a video and write them to a coords .h5 file.

    Args:
        video_path: Video to track.
        checkpoint: YOLO pose model. Empty falls back to yolo11n-pose.pt.
        n_individuals: Upper bound on detections per frame, passed to YOLO as
            max_det.

    Returns:
        Path to the written coords file.

    Raises:
        FileNotFoundError: If video_path is empty or does not exist.
    """
    if not video_path or not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path!r}")

    # lazy: 490ms, and keeping it here is what lets the rest of the batch
    # pipeline be tested without a GPU or model weights
    from ultralytics import YOLO

    model = YOLO(checkpoint) if checkpoint else YOLO("yolo11n-pose.pt")

    results = model.track(
        source=video_path,
        stream=True,
        max_det=n_individuals,
    )

    video_buffers: Dict = {}
    for result in results:
        _accumulate_keypoints(result, video_buffers)

    coords_data = _buffers_to_coords_data(video_buffers)
    return save_coords_to_h5(coords_data, video_path)


def _accumulate_keypoints(result, video_buffers: Dict) -> None:
    """Append one tracking result to the per-video buffers, keyed by source."""
    vid = result.path
    frame = result.frame
    if vid not in video_buffers:
        video_buffers[vid] = {"pts": [], "conf": [], "ind": [], "node": []}
    buf = video_buffers[vid]

    kp = result.keypoints
    if kp is None:
        return
    xy = kp.xy
    conf = (
        kp.conf
        if kp.conf is not None
        else torch.ones(xy.shape[:2], device=xy.device)
    )
    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int()
    else:
        track_ids = torch.arange(xy.shape[0], device=xy.device)

    P, K, _ = xy.shape
    xy_flat = xy.reshape(-1, 2)
    conf_flat = conf.reshape(-1)
    node_flat = torch.tile(torch.arange(K, device=xy.device), (P,))
    ind_flat = torch.repeat_interleave(track_ids, K)
    frame_col = torch.full((P * K,), frame, device=xy.device)
    pts = torch.stack((frame_col, xy_flat[:, 1], xy_flat[:, 0]), dim=1)

    buf["pts"].append(pts)
    buf["conf"].append(conf_flat)
    buf["ind"].append(ind_flat)
    buf["node"].append(node_flat)


def _buffers_to_coords_data(video_buffers: Dict) -> Dict:
    """Pivot the accumulated buffers into {individual: {"x", "y", "ci"}}."""
    coords_data: Dict = {}
    for vid_path, buf in video_buffers.items():
        if not buf["pts"]:
            continue
        pts_np = torch.cat(buf["pts"]).cpu().numpy()
        conf_np = torch.cat(buf["conf"]).cpu().numpy()
        ind_np = torch.cat(buf["ind"]).cpu().numpy()
        node_np = torch.cat(buf["node"]).cpu().numpy()
        df = pd.DataFrame({"frame": pts_np[:, 0].astype(int), "y": pts_np[:, 1],
                            "x": pts_np[:, 2], "ci": conf_np,
                            "ind": ind_np, "node": node_np})
        n_nodes = int(node_np.max()) + 1
        n_frames = int(pts_np[:, 0].max()) + 1
        for ind_id in df.ind.unique():
            sub = df[df.ind == ind_id]
            empty = np.full((n_nodes, n_frames), np.nan)
            for datum in ["x", "y", "ci"]:
                arr = empty.copy()
                pivot = sub.pivot(columns="frame", values=datum, index="node")
                arr_df = pd.DataFrame(arr)
                arr_df.loc[:, pivot.columns] = pivot
                if ind_id not in coords_data:
                    coords_data[ind_id] = {}
                coords_data[ind_id][datum] = arr_df

    return coords_data
