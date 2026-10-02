"""Behaviour decoding: label every frame of a pose file with an ST-GCN decoder."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.registry import LAYOUT_BY_NODE_COUNT, describe_checkpoint
from ..models.st_gcn_aaai18_pylightning_3block import ST_GCN_18
from .dataset import PoseDataset
from .exceptions import CheckpointError, PoseFormatError
from .io import convert_dlc_to_ctvm, read_dlc, read_poser_coords, read_sleap
from .preprocessing import mirror_y

log = logging.getLogger(__name__)


def decoder_settings(checkpoint: dict) -> dict:
    """Read a decoder's preprocessing settings, filling in what it did not store.

    Every fallback is a fixed bug and must not regress (FR-B8):

        transform    data_cfg["transform"]; ["center", "align", "pad"] when falsy
        center_node  graph_cfg["center_node"], else graph_cfg["center"], else 0
        T2           data_cfg["T2"], else 100
        head_node    data_cfg["head"], else 0
        num_class    hyper_parameters["num_class"], else from fcn.weight's shape
        mirror_y     False if data_cfg["preprocess_frame"] is true, else True

    Args:
        checkpoint: A checkpoint dict, as torch.load returns it.

    Returns:
        Keys transform, center_node, T2, head_node, num_class and mirror_y.
    """
    hp = checkpoint.get("hyper_parameters", {}) or {}
    data_cfg = hp.get("data_cfg", {}) or {}
    graph_cfg = hp.get("graph_cfg", {}) or {}
    return {
        # A stored None means "not recorded", not "skip". Skipping leaves the
        # decoder with data unlike its training data, and every frame comes out
        # as one class.
        "transform": data_cfg.get("transform") or ["center", "align", "pad"],
        # Released checkpoints name it "center", `poser model repair` writes
        # "center_node". Centring on the wrong node took accuracy from 0.951 to
        # 0.000 on the test bouts.
        "center_node": int(graph_cfg.get("center_node", graph_cfg.get("center", 0))),
        "T2": int(data_cfg.get("T2", 100)),
        "head_node": int(data_cfg.get("head", 0)),
        "num_class": int(
            hp.get("num_class") or checkpoint["state_dict"]["fcn.weight"].shape[0]
        ),
        # Decoders trained on bouts saw them with y flipped, because bout
        # building flips image y, which points down (classification_data_to_bouts,
        # preprocess_bouts). Decoders trained per frame on *_pose.npy saw raw
        # image coordinates. Skipping the flip mirrored every fish, so left turns
        # read as right: accuracy 0.081, against 0.935 with it.
        "mirror_y": not data_cfg.get("preprocess_frame", False),
    }


class Inference:
    """Decode behaviour from a pose file: load the poses and decoder, label frames."""

    def __init__(self) -> None:
        self._model = None
        self._settings: dict  # from decoder_settings
        self._architecture: dict  # from describe_checkpoint
        self._checkpoint: Path
        self._device: torch.device
        self._pose: np.ndarray # (C, T, V, M)
        self._pose_path: Path

    def model_load(self, path: Path, device: torch.device) -> dict:
        """Load an ST-GCN decoder checkpoint onto device.

        The architecture is read from the weights, via
        models.registry.describe_checkpoint, not from stored metadata: four of
        the eight published checkpoints saved none (FR-B7). A checkpoint with no
        stored data_cfg still loads, with a warning, because its preprocessing
        falls back to defaults that may not match its training.

        Args:
            path: A .ckpt file written by training.
            device: Where to run, normally core.settings.resolve_device().

        Returns:
            The decoder's preprocessing settings, from decoder_settings, for the
            panel to show.

        Raises:
            FileNotFoundError: If path does not exist.
            CheckpointError: If the checkpoint cannot be rebuilt, for example
                because it pickles a WindowsPath (FR-B9).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Decoder checkpoint not found: {path}")

        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except NotImplementedError as exc:
            raise CheckpointError(
                f"{path.name} was saved on Windows and stores a Windows file path, "
                "which cannot be loaded on this system."
            ) from exc

        architecture = describe_checkpoint(checkpoint)
        if architecture["layout"] is None:
            raise CheckpointError(
                f"{path.name} was trained on a {architecture['num_nodes']}-node "
                "skeleton, and no layout in poser.models.graph defines one. Known "
                f"node counts: {sorted(LAYOUT_BY_NODE_COUNT)}."
            )

        hp = checkpoint.get("hyper_parameters", {}) or {}
        if "data_cfg" not in hp:
            log.warning(
                "%s stores no data_cfg, so its preprocessing falls back to defaults "
                "and its labels may be wrong. Run 'poser model repair' to embed "
                "the settings it was trained with.",
                path.name,
            )
        settings = decoder_settings(checkpoint)

        model = ST_GCN_18.load_from_checkpoint(
            str(path),
            map_location=device,
            weights_only=False,
            # The network's shape, read from the weights (FR-B7).
            in_channels=architecture["in_channels"],
            num_class=architecture["num_class"],
            graph_cfg={"layout": architecture["layout"]},
        )

        self._model = model.eval().to(device)
        self._settings = settings
        self._architecture = architecture
        self._checkpoint = path
        self._device = device
        return settings

    def input_load(
        self, path: Path | None = None, coords_data: dict | None = None
    ) -> str:
        """Load pose data as a (C, T, V, M) array.

        Tried in order (FR-B2): *_pose.npy, DeepLabCut .h5/.csv, SLEAP .h5,
        PoseR-native coords .h5, then coords_data already loaded by the Data
        panel.

        Args:
            path: A pose file, or None to use coords_data alone.
            coords_data: {individual: {"x", "y", "ci"}}, each (V, T).

        Returns:
            Which form was used, for the panel to report.

        Raises:
            PoseFormatError: If no form gives a pose array.
        """
        pose_path = Path(path) if path else None
        pose, form, file_error = None, "", None

        if pose_path is not None and pose_path.exists():
            try:
                pose, form = self._read_pose_file(pose_path)
            except PoseFormatError as exc:
                file_error = exc

        if pose is None and coords_data:
            pose = self._coords_to_ctvm(coords_data)
            form = "coords loaded by the Data panel"

        if pose is None:
            raise PoseFormatError(
                f"Could not build a pose array. Pose path: {path or '(none)'}\n"
                "Supported inputs: *_pose.npy, DeepLabCut .h5 / .csv, SLEAP .h5, "
                "PoseR-native coords .h5, or a file loaded in the Data panel."
            ) from file_error

        if pose.ndim == 3:  # (C, T, V) with no individual axis
            pose = pose[..., np.newaxis]
        self._pose = pose
        self._pose_path = pose_path
        return form

    @property
    def n_frames(self) -> int:
        """Frames in the loaded pose, for progress."""
        return self._pose.shape[1]

    def iter_predictions(self, batch_size: int = 64) -> Iterator[np.ndarray]:
        """Yield predicted labels batch by batch, in frame order.

        Each frame is labelled from the T2-frame window centred on it,
        preprocessed as in training. Concatenated, the batches give one label
        per frame, and their running length is the progress.

        Args:
            batch_size: Windows per forward pass. Lower it if the GPU runs out
                of memory; it does not change the labels.

        Raises:
            RuntimeError: If model_load or input_load has not been called.
            ValueError: If the pose has a different number of body points or
                channels from the decoder's training data.
        """
        if self._model is None or not hasattr(self, "_pose"):
            raise RuntimeError("Call model_load and input_load first.")
        n_channels, n_frames, n_nodes, _ = self._pose.shape
        self._check_pose_fits_decoder(n_channels, n_nodes)

        settings = self._settings
        pose = mirror_y(self._pose) if settings["mirror_y"] else self._pose
        windows = PoseDataset(
            data=pose,
            labels=np.zeros(n_frames, dtype=np.int64),  # unused, but required
            preprocess_frame=True,  # one window per frame, centred on it
            window_size=settings["T2"],
            T=settings["T2"],
            transform=settings["transform"],
            center_node=settings["center_node"],
            head_node=settings["head_node"],
            num_class=settings["num_class"],
            C=n_channels,
            augmentation=None,
        )
        loader = DataLoader(windows, batch_size=batch_size, shuffle=False)
        for batch, _ in loader:
            with torch.no_grad():
                scores = self._model(batch.to(self._device))
            yield scores.argmax(dim=-1).cpu().numpy()

    def predictions_save(self, predictions: np.ndarray) -> Path:
        """Write <stem>_predictions.npy and .csv beside the pose file (FR-B4).

        The .csv has columns frame,predicted_label. A *_pose.npy file loses its
        "_pose", so rec_pose.npy gives rec_predictions.npy.

        Returns:
            Path to the .npy file.

        Raises:
            ValueError: If the pose came from memory, with no file to save beside.
        """
        if self._pose_path is None:
            raise ValueError(
                "This pose came from the Data panel's memory, so there is no pose "
                "file to save the predictions beside."
            )
        stem = self._pose_path.stem
        if self._pose_path.name.endswith("_pose.npy"):
            stem = stem[: -len("_pose")]
        npy_path = self._pose_path.parent / f"{stem}_predictions.npy"
        np.save(npy_path, predictions)

        with open(npy_path.with_suffix(".csv"), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame", "predicted_label"])
            writer.writerows(enumerate(predictions.tolist()))
        return npy_path

    def _check_pose_fits_decoder(self, n_channels: int, n_nodes: int) -> None:
        """Fail with a plain message, not a torch matrix-size error, on a mismatch."""
        expected_nodes = self._architecture["num_nodes"]
        expected_channels = self._architecture["in_channels"]
        if n_nodes != expected_nodes:
            raise ValueError(
                f"The pose has {n_nodes} body points, but {self._checkpoint.name} "
                f"was trained on {expected_nodes}. Pick a decoder trained on this "
                "skeleton."
            )
        if n_channels != expected_channels:
            raise ValueError(
                f"The pose has {n_channels} channels per body point, but "
                f"{self._checkpoint.name} expects {expected_channels}."
            )

    @staticmethod
    def _read_pose_file(path: Path) -> tuple[np.ndarray, str]:
        """Read a pose file as (C, T, V, M) or (C, T, V), with the form it was.

        Raises:
            PoseFormatError: If no reader recognises the file. The message
                names what each reader rejected it for.
        """
        if path.name.endswith("_pose.npy"):
            return np.load(path).astype(np.float32), "*_pose.npy"

        to_ctvm = Inference._coords_to_ctvm
        readers = (
            ("DeepLabCut", lambda: convert_dlc_to_ctvm(path).astype(np.float32)),
            ("DeepLabCut", lambda: to_ctvm(read_dlc(path))),
            ("SLEAP", lambda: to_ctvm(read_sleap(path))),
            ("PoseR-native", lambda: to_ctvm(read_poser_coords(path))),
        )
        failures = []
        for form, read in readers:
            try:
                return read(), form
            except Exception as exc:
                # Format probe: any parse failure just means "not this format".
                failures.append(f"{form}: {type(exc).__name__}: {exc}")
        raise PoseFormatError(
            f"{path.name} is not a readable pose file. Tried:\n  "
            + "\n  ".join(failures)
        )

    @staticmethod
    def _coords_to_ctvm(coords_data: dict) -> np.ndarray:
        """Convert {individual: {"x", "y", "ci"}} into a (C=3, T, V, M) float32 array.

        Values may be DataFrames or arrays, each (V, T); a 1-D array is a single
        node. NaN becomes 0.
        """
        individuals = list(coords_data)
        first = np.array(coords_data[individuals[0]]["x"], dtype=np.float32)
        if first.ndim == 1:
            first = first[np.newaxis, :]
        n_nodes, n_frames = first.shape

        ctvm = np.zeros((3, n_frames, n_nodes, len(individuals)), dtype=np.float32)
        for m, individual in enumerate(individuals):
            for c, key in enumerate(("x", "y", "ci")):
                values = np.array(coords_data[individual][key], dtype=np.float32)
                values = np.nan_to_num(values)
                if values.ndim == 1:
                    values = values[np.newaxis, :]
                ctvm[c, :, :, m] = values.T  # (V, T) -> (T, V)
        return ctvm
