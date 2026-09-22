"""
Batch pipeline — run pose estimation or behaviour decoding over many files.

Feature 1: load and analyse multiple files at once.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .bout_detection import orthogonal_variance
from .io import read_coords
from .pose_estimation import estimate_poses_from_video
from .preprocessing import preprocess_bouts
from .schemas.batch import BatchMode, BatchResult

log = logging.getLogger(__name__)


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
    mode: str = BatchMode.BEHAVIOUR.value
    checkpoint: str = ""
    config = None           # TrainingConfig or path string
    output_dir: str = ""
    n_individuals: int = 1
    progress_callback: Optional[Callable[[int, int], None]] = None


    def run(self) -> List[BatchResult]:
        """Execute the batch job.

        Iterates through *pose_files* / *video_files*, dispatches to the
        appropriate inference function, writes per-file outputs, and
        generates a ``batch_manifest.csv``.

        Returns
        -------
        list of :class:`BatchResult`
        """
        Path(self.output_dir or ".").mkdir(parents=True, exist_ok=True)
        mode = BatchMode(self.mode)
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
                if mode is BatchMode.POSE_ESTIMATION:
                    out = estimate_poses_from_video(
                        video_path, self.checkpoint, self.n_individuals
                    )
                else:
                    out = self._run_behaviour_decode(pose_path, video_path, i)

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
                log.error("Error processing %s: %s", pose_path, exc)

        if self.progress_callback:
            self.progress_callback(total, total)

        self._write_manifest(results)
        return results

    def _run_behaviour_decode(self, pose_path: str, video_path: str, idx: int) -> str:
        """Run behaviour decoding on *pose_path* and save classification .h5."""
        coords_data = read_coords(pose_path)

        # Use first individual
        ind_key = next(iter(coords_data))
        data = coords_data[ind_key]

        x = np.array(data["x"]) if hasattr(data["x"], "__array__") else data["x"]
        y = np.array(data["y"]) if hasattr(data["y"], "__array__") else data["y"]
        ci_arr = np.array(data["ci"]) if hasattr(data["ci"], "__array__") else data["ci"]

        # Resolve config
        cfg = self.config
        if isinstance(cfg, str):
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
        from ..models.registry import ModelRegistry  # noqa: F401  broken, fixed in D6

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
        pose_file = Path(pose_path)
        out_dir = Path(self.output_dir) if self.output_dir else pose_file.parent
        out_path = out_dir / f"{pose_file.stem}_predictions.npy"
        np.save(out_path, predictions)
        return str(out_path)

    def _write_manifest(self, results: List[BatchResult]) -> None:
        if not self.output_dir:
            return
        path = Path(self.output_dir) / "batch_manifest.csv"
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
        log.info("Manifest written to %s", path)
