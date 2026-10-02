"""Behaviour decoding: label every frame of a pose file with an ST-GCN decoder."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch


def decoder_settings(checkpoint: dict) -> dict:
    """Read a decoder's preprocessing settings, filling in what it did not store.

    Every fallback is a fixed bug and must not regress (FR-B8):

        transform    data_cfg["transform"]; ["center", "align", "pad"] when falsy
        center_node  graph_cfg["center_node"], else graph_cfg["center"], else 0
        T2           data_cfg["T2"], else 100
        head_node    data_cfg["head"], else 0
        num_class    hyper_parameters["num_class"], else from fcn.weight's shape

    Args:
        checkpoint: A checkpoint dict, as torch.load returns it.

    Returns:
        Keys transform, center_node, T2, head_node and num_class.
    """
    raise NotImplementedError


class Inference:
    """Decode behaviour from a pose file: load the poses and decoder, label frames."""

    def __init__(self) -> None:
        self._model = None
        self._settings: dict  # from decoder_settings
        self._checkpoint: Path
        self._pose: np.ndarray # (C, T, V, M)
        self._pose_path: Path

    def model_load(self, path: Path, device: torch.device) -> None:
        """Load an ST-GCN decoder checkpoint onto device.

        The architecture is read from the weights, via
        models.registry.describe_checkpoint, not from stored metadata: four of
        the eight published checkpoints saved none (FR-B7).

        Args:
            path: A .ckpt file written by training.
            device: Where to run, normally core.settings.resolve_device().

        Raises:
            FileNotFoundError: If path does not exist.
            CheckpointError: If the checkpoint cannot be rebuilt, for example
                because it pickles a WindowsPath (FR-B9).
        """
        raise NotImplementedError

    def input_load(
        self, path: Path | None = None, coords_data: dict | None = None
    ) -> None:
        """Load pose data as a (C, T, V, M) array.

        Tried in order (FR-B2): *_pose.npy, DeepLabCut .h5/.csv, SLEAP .h5,
        PoseR-native coords .h5, then coords_data already loaded by the Data
        panel.

        Args:
            path: A pose file, or None to use coords_data alone.
            coords_data: {individual: {"x", "y", "ci"}}, each (V, T).

        Raises:
            PoseFormatError: If no form gives a pose array.
        """
        raise NotImplementedError

    def iter_predictions(self) -> Iterator[np.ndarray]:
        """Yield predicted labels batch by batch, in frame order.

        Each frame is labelled from the T2-frame window centred on it,
        preprocessed as in training. Concatenated, the batches give one label
        per frame, and their running length is the progress.

        Raises:
            RuntimeError: If model_load or input_load has not been called.
        """
        raise NotImplementedError

    def predictions_save(self, predictions: np.ndarray) -> Path:
        """Write <stem>_predictions.npy and .csv beside the pose file (FR-B4).

        The .csv has columns frame,predicted_label.

        Returns:
            Path to the .npy file.
        """
        raise NotImplementedError
