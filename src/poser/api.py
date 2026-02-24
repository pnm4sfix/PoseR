"""
api.py
~~~~~~
High-level :class:`PoseR` facade for notebook / script use.

This is the primary Python API for interacting with PoseR programmatically,
without the napari GUI or CLI.

Example
-------
::

    from poser import PoseR

    p = PoseR(config="decoder_config.yml")

    # Load and analyse a single pose file
    coords = p.load("my_recording.h5")
    bouts  = p.detect_bouts(coords, individual="ind1")
    preds  = p.predict_behaviours(bouts, checkpoint="model.ckpt")

    # Batch
    results = p.batch(pose_files=["a.h5", "b.h5"], checkpoint="model.ckpt")

    # Train (fully external — launches CLI subprocess)
    p.train(files=["labelled.h5"], epochs=100)
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

log = logging.getLogger(__name__)


class PoseR:
    """Programmatic entry-point for PoseR.

    Parameters
    ----------
    config:
        Path to a ``decoder_config.yml`` or a
        :class:`~poser.training.config.TrainingConfig` instance.
        If ``None``, a default configuration is used.
    device:
        Inference device — ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``.
    """

    def __init__(
        self,
        config: Optional[Union[str, Path, "TrainingConfig"]] = None,
        device: str = "auto",
    ):
        from poser.training.config import TrainingConfig

        if config is None:
            self._config = TrainingConfig()
        elif isinstance(config, TrainingConfig):
            self._config = config
        else:
            self._config = TrainingConfig.from_yaml(config)

        self._device = device

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> "TrainingConfig":
        return self._config

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(
        self,
        pose_file: str | Path,
        *,
        confidence_threshold: Optional[float] = None,
        bodypoints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load a pose file and return a ``coords_data`` dict.

        Supports DLC HDF5, SLEAP HDF5, and PoseR native HDF5.

        Returns
        -------
        coords_data : dict
            ``{individual_key: {"x": ndarray, "y": ndarray, "ci": ndarray}}``
        """
        from poser.core.io import read_coords

        thresh = confidence_threshold or self._config.data.confidence_threshold
        return read_coords(
            pose_file,
            confidence_threshold=thresh,
            bodypoints=bodypoints,
        )

    # ------------------------------------------------------------------
    # Bout detection
    # ------------------------------------------------------------------

    def detect_bouts(
        self,
        coords_data: Dict[str, Any],
        *,
        individual: str = "ind1",
        method: str = "orthogonal",
        fps: Optional[int] = None,
        center_node: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect locomotion bouts from *coords_data*.

        Parameters
        ----------
        coords_data:
            Output of :meth:`load` or ``read_coords``.
        individual:
            Key within *coords_data* to analyse.
        method:
            ``"orthogonal"`` (default) or ``"egocentric"``.
        fps:
            Overrides ``config.data.fps``.
        center_node:
            Overrides skeleton centre node index.

        Returns
        -------
        bouts : list of dicts
            Each dict has keys ``start``, ``end``, ``label``, ``coords``.
        """
        from poser.core.bout_detection import orthogonal_variance, egocentric_variance
        from poser.skeletons.registry import get_skeleton

        fps = fps or self._config.data.fps
        skel = get_skeleton(self._config.model.layout)
        cnode = center_node if center_node is not None else skel.center_node

        ind_data = coords_data[individual]
        # Build (T, V, 2) points array
        x = np.array(ind_data["x"])
        y = np.array(ind_data["y"])
        ci = np.array(ind_data["ci"])
        points = np.stack([x, y], axis=-1)  # (T, V, 2)

        if method == "egocentric":
            bouts, *_ = egocentric_variance(
                points, center_node=cnode, fps=fps,
                n_nodes=skel.num_nodes,
            )
        else:
            bouts, *_ = orthogonal_variance(
                points, center_node=cnode, fps=fps,
                n_nodes=skel.num_nodes,
            )

        return bouts

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        coords_data: Dict[str, Any],
        bouts: List[Dict[str, Any]],
        *,
        individual: str = "ind1",
    ) -> np.ndarray:
        """Preprocess *bouts* into model-ready tensor.

        Returns
        -------
        X : np.ndarray, shape ``(N, C, T2, V, M)``
        """
        from poser.core.preprocessing import preprocess_bouts

        ind_data = coords_data[individual]
        x = np.array(ind_data["x"])
        y = np.array(ind_data["y"])
        ci = np.array(ind_data["ci"])
        egocentric = np.stack([x, y], axis=-1)  # (T, V, 2)

        dcfg = self._config.data
        mcfg = self._config.model

        padded, _ = preprocess_bouts(
            egocentric=egocentric,
            ci=ci,
            bouts=bouts,
            C=dcfg.C,
            T=dcfg.T,
            T2=dcfg.T2,
            fps=dcfg.fps,
            denominator=dcfg.denominator,
        )
        return padded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        pose_file: Optional[str | Path] = None,
        *,
        coords_data: Optional[Dict[str, Any]] = None,
        checkpoint: str | Path,
        individual: str = "ind1",
    ) -> Dict[int, int]:
        """Run behaviour inference.

        Either *pose_file* or *coords_data* must be provided.

        Returns
        -------
        predictions : dict
            ``{frame_index: predicted_label}``
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from poser.models.registry import load_model

        if coords_data is None:
            if pose_file is None:
                raise ValueError("Provide either pose_file or coords_data.")
            coords_data = self.load(pose_file)

        bouts = self.detect_bouts(coords_data, individual=individual)
        if not bouts:
            log.warning("No bouts detected — returning empty predictions.")
            return {}

        X = self.preprocess(coords_data, bouts, individual=individual)
        mcfg = self._config.model

        model = load_model(
            mcfg.architecture,
            checkpoint=checkpoint,
            num_class=mcfg.num_class,
            num_nodes=mcfg.num_nodes,
            in_channels=mcfg.in_channels,
            layout=mcfg.layout,
        )
        model.eval()

        tensor = torch.from_numpy(X).float()  # (N, C, T2, V, M)
        dl = DataLoader(TensorDataset(tensor), batch_size=32, shuffle=False)

        all_preds: List[int] = []
        with torch.no_grad():
            for (batch,) in dl:
                logits = model(batch)
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)

        return {b["start"]: p for b, p in zip(bouts, all_preds)}

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def batch(
        self,
        pose_files: List[str | Path],
        *,
        checkpoint: Optional[str | Path] = None,
        mode: str = "behaviour",
        output_dir: str | Path = "batch_output",
        n_individuals: int = 1,
    ) -> List[Any]:
        """Batch-process multiple pose files.

        Returns
        -------
        results : list of BatchResult
        """
        from poser.core.batch import BatchJob

        job = BatchJob(
            pose_files=[Path(f) for f in pose_files],
            mode=mode,
            checkpoint=Path(checkpoint) if checkpoint else None,
            config=self._config,
            output_dir=Path(output_dir),
            n_individuals=n_individuals,
        )
        return job.run()

    # ------------------------------------------------------------------
    # Training (external subprocess)
    # ------------------------------------------------------------------

    def train(
        self,
        files: List[str | Path],
        *,
        config_path: Optional[str | Path] = None,
        epochs: Optional[int] = None,
        run_name: Optional[str] = None,
        blocking: bool = True,
    ) -> Optional[subprocess.Popen]:
        """Launch training as a ``poser train`` CLI subprocess.

        This keeps the napari GUI (or calling notebook) responsive.

        Parameters
        ----------
        files:
            Labelled HDF5 classification files.
        config_path:
            YAML config to pass to the CLI.  If ``None``, saves the current
            :attr:`config` to a temp file.
        epochs:
            Override ``trainer.max_epochs``.
        run_name:
            Override ``run_name``.
        blocking:
            If ``True`` (default), wait for the subprocess to finish.

        Returns
        -------
        proc : subprocess.Popen or None
            ``None`` when ``blocking=True`` (training has finished).
        """
        import tempfile

        cfg_path = config_path
        if cfg_path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".yml", delete=False, mode="w"
            )
            self._config.to_yaml(tmp.name)
            cfg_path = tmp.name

        cmd = [
            sys.executable, "-m", "poser.cli.main",
            "train",
            "--config", str(cfg_path),
            *[str(f) for f in files],
        ]
        if epochs:
            cmd += ["--epochs", str(epochs)]
        if run_name:
            cmd += ["--name", run_name]

        log.info("Launching training subprocess: %s", cmd)
        proc = subprocess.Popen(cmd)

        if blocking:
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Training failed with exit code {proc.returncode}.")
            return None

        return proc

    # ------------------------------------------------------------------
    # Skeleton helpers
    # ------------------------------------------------------------------

    def list_skeletons(self) -> List[str]:
        """Return all registered skeleton names."""
        from poser.skeletons.registry import list_skeletons
        return list_skeletons()

    def get_skeleton(self, name: str):
        """Return a :class:`~poser.skeletons.base.BaseSkeletonSpec` by name."""
        from poser.skeletons.registry import get_skeleton
        return get_skeleton(name)

    def __repr__(self) -> str:
        return (
            f"PoseR(layout={self._config.model.layout!r}, "
            f"num_class={self._config.model.num_class}, "
            f"device={self._device!r})"
        )
