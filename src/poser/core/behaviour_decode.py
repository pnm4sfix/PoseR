"""Detect bouts in a pose file and classify them with a trained model.

Extracted from BatchJob._run_behaviour_decode. The points-array layout below
is preserved verbatim from that method and is known to disagree with itself;
see the Tier-2 notes in REFACTOR_PLAN Phase 3.2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .bout_detection import orthogonal_variance
from .exceptions import CheckpointError
from .io import read_coords
from .preprocessing import preprocess_bouts
from .schemas.training import DataConfig, ModelConfig, TrainingConfig
from .settings import settings

log = logging.getLogger(__name__)


def decode_behaviours(
    pose_path: str,
    checkpoint: str = "",
    config: Optional[TrainingConfig] = None,
    output_dir: str = "",
) -> str:
    """Classify the bouts in one pose file and save the predictions.

    Args:
        pose_path: Pose file to read, in any format core.io recognises.
        checkpoint: Trained classifier checkpoint. Required.
        config: Supplies the bout detection and preprocessing parameters.
            None uses the DataConfig and ModelConfig defaults.
        output_dir: Where to write the .npy. Empty writes beside pose_path.

    Returns:
        Path to the written predictions, or an empty string when no bouts
        were detected.

    Raises:
        CheckpointError: If checkpoint is empty.
    """
    data_cfg = config.data if config else DataConfig()
    model_cfg = config.model if config else ModelConfig()

    coords_data = read_coords(pose_path)
    x, y, ci_arr = _first_individual(coords_data)

    n_nodes = x.shape[0] if x.ndim >= 1 else 9
    n_frames = x.shape[1] if x.ndim == 2 else x.shape[0]

    points = _points_from_coords(x, y, n_nodes, n_frames)
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

    egocentric = np.zeros((n_nodes, n_frames, 3))
    egocentric[:, :, 0] = np.arange(n_frames)
    egocentric[:, :, 1] = y if y.ndim == 2 else y.reshape(n_nodes, n_frames)
    egocentric[:, :, 2] = x if x.ndim == 2 else x.reshape(n_nodes, n_frames)

    padded, _ = preprocess_bouts(
        egocentric,
        pd.DataFrame(ci_arr),
        bouts,
        C=data_cfg.C,
        T=data_cfg.T,
        T2=data_cfg.T2,
        fps=data_cfg.fps,
        denominator=data_cfg.denominator,
        # T_method decides how T is derived. Passing it matters: the
        # preprocess_bouts default of "window" computes 2*int(fps/denominator),
        # which is 0 for the DataConfig defaults.
        T_method=data_cfg.T_method,
        head_node=data_cfg.head_node,
    )

    predictions = _predict(padded, model_cfg.architecture, checkpoint)

    pose_file = Path(pose_path)
    out_dir = Path(output_dir) if output_dir else pose_file.parent
    out_path = out_dir / f"{pose_file.stem}_predictions.npy"
    np.save(out_path, predictions)
    return str(out_path)


def _first_individual(coords_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y and ci for the first individual in the file.

    np.array handles both shapes core.io returns: DataFrames from read_dlc,
    ndarrays from read_sleap and read_poser_coords.
    """
    data = coords_data[next(iter(coords_data))]
    return np.array(data["x"]), np.array(data["y"]), np.array(data["ci"])


def _points_from_coords(
    x: np.ndarray, y: np.ndarray, n_nodes: int, n_frames: int
) -> np.ndarray:
    """Build the (n_nodes * n_frames, 3) frame/y/x array bout detection wants."""
    frame_idx = np.tile(np.arange(n_frames), n_nodes)
    y_flat = y.T.reshape(-1) if y.ndim == 2 else y.reshape(-1)
    x_flat = x.T.reshape(-1) if x.ndim == 2 else x.reshape(-1)
    return np.stack([frame_idx, y_flat, x_flat], axis=1).astype(float)


def _predict(padded: np.ndarray, architecture: str, checkpoint: str) -> np.ndarray:
    """Run the classifier over the padded bouts and return one label each."""
    if not checkpoint:
        raise CheckpointError(
            f"Decoding behaviour needs a trained {architecture} checkpoint. "
            "Pass one as BatchJob(checkpoint=...)."
        )

    # circular: poser.models imports _loader, which imports core.augmentation,
    # so this module cannot reach the registry at import time
    from ..models.registry import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(architecture, checkpoint, map_location=str(device))
    model.eval().to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(padded, dtype=torch.float32)),
        batch_size=settings.inference_batch_size,
    )
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            out = model(batch.to(device))
            preds.append(out.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds)
