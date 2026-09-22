"""Run pose estimation or behaviour decoding over many files."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch.utils.data import DataLoader, TensorDataset

from .bout_detection import orthogonal_variance
from .exceptions import CheckpointError
from .io import read_coords
from .pose_estimation import estimate_poses_from_video
from .preprocessing import preprocess_bouts
from .schemas.batch import BatchMode, BatchResult
from .schemas.training import DataConfig, ModelConfig, TrainingConfig

log = logging.getLogger(__name__)


class BatchJob(BaseModel):
    """Configuration for a multi-file batch analysis run.

    Built from CLI arguments and from user code, so the fields are validated
    on construction rather than part way through a long run.

    Attributes:
        video_files: Paired with pose_files by position. Shorter lists are
            padded with empty strings.
        mode: "behaviour" detects and classifies bouts in each pose file;
            "pose_estimation" runs YOLO-pose over each video.
        checkpoint: Model to run. A YOLO pose model under "pose_estimation",
            an ST-GCN classifier under "behaviour".
        config: A TrainingConfig, or a path to a config.yaml.
        output_dir: Where per-file outputs and batch_manifest.csv are written.
        n_individuals: Upper bound on detections per frame, passed to YOLO as
            max_det.
        progress_callback: Called as (completed, total) before each file.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pose_files: List[str] = Field(default_factory=list)
    video_files: List[str] = Field(default_factory=list)
    mode: BatchMode = BatchMode.BEHAVIOUR
    checkpoint: str = ""
    config: Optional[TrainingConfig] = None
    output_dir: str = ""
    n_individuals: int = 1
    progress_callback: Optional[Callable[[int, int], None]] = None

    @field_validator("pose_files", "video_files", mode="before")
    @classmethod
    def _stringify_path_list(cls, value):
        """Accept Path entries, which both callers build with pathlib."""
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return value

    @field_validator("checkpoint", "output_dir", mode="before")
    @classmethod
    def _stringify_path(cls, value):
        """Accept None and Path where a plain string is expected."""
        return "" if value is None else str(value)

    @field_validator("config", mode="before")
    @classmethod
    def _load_config(cls, value):
        """Accept a path to a config.yaml as well as a TrainingConfig."""
        if isinstance(value, (str, Path)):
            return TrainingConfig.from_yaml(value)
        return value


    def run(self) -> List[BatchResult]:
        """Process every input and write batch_manifest.csv beside the outputs.

        A file that fails is recorded as an error result and does not stop the
        rest of the run.

        Returns:
            One BatchResult per input, in input order.
        """
        Path(self.output_dir or ".").mkdir(parents=True, exist_ok=True)
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
                if self.mode is BatchMode.POSE_ESTIMATION:
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
        """Detect bouts in one pose file, classify them, and save the result."""
        coords_data = read_coords(pose_path)

        # Use first individual
        ind_key = next(iter(coords_data))
        data = coords_data[ind_key]

        # np.array handles both shapes core.io returns: DataFrames from
        # read_dlc, ndarrays from read_sleap and read_poser_coords.
        x = np.array(data["x"])
        y = np.array(data["y"])
        ci_arr = np.array(data["ci"])

        data_cfg = self.config.data if self.config else DataConfig()

        n_nodes = x.shape[0] if x.ndim >= 1 else 9

        # Build points array (n_nodes * n_frames, 3)
        n_frames = x.shape[1] if x.ndim == 2 else x.shape[0]
        frame_idx = np.tile(np.arange(n_frames), n_nodes)
        y_flat = y.T.reshape(-1) if y.ndim == 2 else y.reshape(-1)
        x_flat = x.T.reshape(-1) if x.ndim == 2 else x.reshape(-1)
        points = np.stack([frame_idx, y_flat, x_flat], axis=1).astype(float)

        bouts, *_ = orthogonal_variance(
            points,
            center_node=data_cfg.center_node,
            fps=data_cfg.fps,
            n_nodes=n_nodes,
            # TrainingConfig has no amd_threshold field, so this stays the
            # orthogonal_variance default rather than becoming configurable.
            amd_threshold=2.0,
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
            egocentric_nd,
            ci_df,
            bouts,
            C=data_cfg.C,
            T=data_cfg.T,
            T2=data_cfg.T2,
            fps=data_cfg.fps,
            denominator=data_cfg.denominator,
            # T_method decides how T is derived. Passing it matters: the
            # preprocess_bouts default of "window" computes 2*int(fps/
            # denominator), which is 0 for the DataConfig defaults.
            T_method=data_cfg.T_method,
            head_node=data_cfg.head_node,
        )

        # circular: poser.models imports _loader, which imports core.augmentation,
        # so core.batch cannot reach the registry at module level
        from ..models.registry import load_model

        model_cfg = self.config.model if self.config else ModelConfig()
        if not self.checkpoint:
            raise CheckpointError(
                f"Decoding behaviour needs a trained {model_cfg.architecture} "
                "checkpoint. Pass one as BatchJob(checkpoint=...)."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(
            model_cfg.architecture,
            self.checkpoint,
            map_location=str(device),
        )
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
