"""
Batch pipeline — run pose estimation or behaviour decoding over many files.

Feature 1: load and analyse multiple files at once.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class BatchResult:
    """Result from processing one (pose_file, video_file) pair."""

    pose_path: str
    video_path: str
    output_path: str
    status: str       # "ok" | "error"
    error: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class BatchJob:
    """Configuration for a multi-file batch analysis run.

    Parameters
    ----------
    pose_files:
        List of paths to pose estimation files.
    video_files:
        List of paths to paired video files (must match length of
        *pose_files*, or be empty to skip video loading).
    mode:
        ``"pose_estimation"`` — run YOLO-pose on each video and save
        ``_poser_coords.h5`` output.
        ``"behaviour_decode"`` — run ST-GCN/C3D inference on loaded pose
        files and save ``_classification.h5`` output.
    checkpoint:
        Path to the model checkpoint to use for inference.
    config:
        :class:`poser.training.config.TrainingConfig` instance (or the path
        to a ``config.yaml`` file).
    output_dir:
        Directory to write per-file outputs and the batch manifest.
    n_individuals:
        Expected number of individuals per frame (for YOLO ``max_det``).
    progress_callback:
        Optional callable ``(completed: int, total: int)`` for UI progress
        reporting.
    """

    pose_files: List[str] = field(default_factory=list)
    video_files: List[str] = field(default_factory=list)
    mode: str = "behaviour_decode"
    checkpoint: str = ""
    config = None           # TrainingConfig or path string
    output_dir: str = ""
    n_individuals: int = 1
    progress_callback: Optional[Callable[[int, int], None]] = None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> List[BatchResult]:
        """Execute the batch job.

        Iterates through *pose_files* / *video_files*, dispatches to the
        appropriate inference function, writes per-file outputs, and
        generates a ``batch_manifest.csv``.

        Returns
        -------
        list of :class:`BatchResult`
        """
        os.makedirs(self.output_dir or ".", exist_ok=True)
        total = len(self.pose_files)
        results: List[BatchResult] = []

        # Pad video list if shorter
        video_files = list(self.video_files)
        while len(video_files) < total:
            video_files.append("")

        for i, (pose_path, video_path) in enumerate(zip(self.pose_files, video_files)):
            if self.progress_callback:
                self.progress_callback(i, total)

            try:
                if self.mode == "pose_estimation":
                    out = self._run_pose_estimation(pose_path, video_path, i)
                elif self.mode == "behaviour_decode":
                    out = self._run_behaviour_decode(pose_path, video_path, i)
                else:
                    raise ValueError(f"Unknown mode: {self.mode!r}")

                results.append(
                    BatchResult(
                        pose_path=pose_path,
                        video_path=video_path,
                        output_path=out,
                        status="ok",
                    )
                )
            except Exception as exc:
                results.append(
                    BatchResult(
                        pose_path=pose_path,
                        video_path=video_path,
                        output_path="",
                        status="error",
                        error=str(exc),
                    )
                )
                print(f"  [BatchJob] Error processing {pose_path}: {exc}")

        if self.progress_callback:
            self.progress_callback(total, total)

        self._write_manifest(results)
        return results

    # ------------------------------------------------------------------
    # Mode-specific helpers
    # ------------------------------------------------------------------

    def _run_pose_estimation(self, pose_path: str, video_path: str, idx: int) -> str:
        """Run YOLO-pose on *video_path* and save a coords .h5 file."""
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path!r}")

        from ultralytics import YOLO
        import torch
        import numpy as np
        from .io import save_coords_to_h5

        model = YOLO(self.checkpoint) if self.checkpoint else YOLO("yolo11n-pose.pt")

        results = model.track(
            source=video_path,
            stream=True,
            max_det=self.n_individuals,
        )

        video_buffers: Dict = {}
        for result in results:
            vid = result.path
            frame = result.frame
            if vid not in video_buffers:
                video_buffers[vid] = {"pts": [], "conf": [], "ind": [], "node": []}
            buf = video_buffers[vid]

            kp = result.keypoints
            if kp is None:
                continue
            xy = kp.xy
            conf = kp.conf if kp.conf is not None else torch.ones(xy.shape[:2], device=xy.device)
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

        # Convert to coords_data and save
        import pandas as pd
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

        out_path = save_coords_to_h5(coords_data, video_path)
        return out_path

    def _run_behaviour_decode(self, pose_path: str, video_path: str, idx: int) -> str:
        """Run behaviour decoding on *pose_path* and save classification .h5."""
        from .io import read_coords
        from .bout_detection import orthogonal_variance, egocentric_variance
        from .preprocessing import preprocess_bouts

        coords_data = read_coords(pose_path)

        # Use first individual
        ind_key = next(iter(coords_data))
        data = coords_data[ind_key]

        import numpy as np
        import pandas as pd

        x = np.array(data["x"]) if hasattr(data["x"], "__array__") else data["x"]
        y = np.array(data["y"]) if hasattr(data["y"], "__array__") else data["y"]
        ci_arr = np.array(data["ci"]) if hasattr(data["ci"], "__array__") else data["ci"]

        # Resolve config
        cfg = self.config
        if isinstance(cfg, str):
            import yaml
            with open(cfg) as f:
                cfg_dict = yaml.safe_load(f)
        elif cfg is not None:
            cfg_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else vars(cfg)
        else:
            cfg_dict = {}

        fps = cfg_dict.get("fps", 30.0)
        n_nodes = x.shape[0] if x.ndim >= 1 else 9
        center_node = cfg_dict.get("center_node", 0)

        # Build points array (n_nodes * n_frames, 3)
        n_frames = x.shape[1] if x.ndim == 2 else x.shape[0]
        frame_idx = np.tile(np.arange(n_frames), n_nodes)
        y_flat = y.T.reshape(-1) if y.ndim == 2 else y.reshape(-1)
        x_flat = x.T.reshape(-1) if x.ndim == 2 else x.reshape(-1)
        points = np.stack([frame_idx, y_flat, x_flat], axis=1).astype(float)

        bouts, gauss, threshold, _ = orthogonal_variance(
            points, center_node=center_node, fps=fps, n_nodes=n_nodes,
            amd_threshold=cfg_dict.get("amd_threshold", 2.0),
        )

        if not bouts:
            return ""

        # Preprocess bouts
        import pandas as pd
        egocentric_nd = np.zeros((n_nodes, n_frames, 3))
        egocentric_nd[:, :, 0] = np.arange(n_frames)
        egocentric_nd[:, :, 1] = y if y.ndim == 2 else y.reshape(n_nodes, n_frames)
        egocentric_nd[:, :, 2] = x if x.ndim == 2 else x.reshape(n_nodes, n_frames)
        ci_df = pd.DataFrame(ci_arr)

        padded, _ = preprocess_bouts(
            egocentric_nd, ci_df, bouts,
            fps=fps,
            T2=cfg_dict.get("T2", 50),
            denominator=cfg_dict.get("denominator", 8),
        )

        # Inference
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from ..models.registry import ModelRegistry

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ModelRegistry.load(self.checkpoint)
        model.eval().to(device)

        tensor_data = torch.tensor(padded, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor_data), batch_size=16)
        preds = []
        with torch.no_grad():
            for (batch,) in loader:
                out = model(batch.to(device))
                preds.append(out.argmax(dim=1).cpu().numpy())

        predictions = np.concatenate(preds)

        # Save outputs
        out_dir = self.output_dir or os.path.dirname(pose_path)
        stem = Path(pose_path).stem
        out_path = os.path.join(out_dir, f"{stem}_predictions.npy")
        np.save(out_path, predictions)
        return out_path

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _write_manifest(self, results: List[BatchResult]) -> None:
        if not self.output_dir:
            return
        path = os.path.join(self.output_dir, "batch_manifest.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["pose_file", "video_file", "output", "status", "error"]
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "pose_file": r.pose_path,
                        "video_file": r.video_path,
                        "output": r.output_path,
                        "status": r.status,
                        "error": r.error,
                    }
                )
        print(f"[BatchJob] Manifest written to {path}")
