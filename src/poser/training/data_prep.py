"""
training/data_prep.py
~~~~~~~~~~~~~~~~~~~~~
High-level helper that turns a collection of labelled pose files into a
:class:`~torch.utils.data.DataLoader` pair (train / val) ready to pass to
:class:`~poser.training.trainer.PoseRTrainer`.

This replaces the scattered ``prepare_data`` / ``preprocess_bouts`` calls
spread across ``_widget.py``.

Usage
-----
::

    from poser.training import TrainingConfig
    from poser.training.data_prep import prepare_dataset

    cfg = TrainingConfig.from_yaml("decoder_config.yml")
    train_dl, val_dl = prepare_dataset(classification_h5_files, cfg)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from poser.core.io import read_classification_h5
from poser.core.preprocessing import classification_data_to_bouts
from poser.core.dataset import PoseDataset
from poser.training.config import TrainingConfig

log = logging.getLogger(__name__)


def prepare_dataset(
    classification_files: List[str | Path],
    config: TrainingConfig,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build ``(train_loader, val_loader)`` from *classification_files*.

    Parameters
    ----------
    classification_files:
        List of paths to HDF5 files written by ``save_to_h5`` / the
        annotation panel.  Each file contains poses + behaviour labels.
    config:
        A :class:`~poser.training.config.TrainingConfig` instance.
    num_workers:
        Workers for :class:`torch.utils.data.DataLoader`.
    pin_memory:
        Pin memory for CUDA training.

    Returns
    -------
    train_loader, val_loader
        ``val_loader`` is ``None`` when ``config.trainer.val_split == 0``.
    """
    dcfg = config.data
    tcfg = config.trainer
    mcfg = config.model
    aug = config.augmentation

    # Accumulate all bouts and labels across files
    all_bouts: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for fpath in classification_files:
        fpath = Path(fpath)
        try:
            classification_data = read_classification_h5(fpath)
        except Exception as exc:
            log.warning("Skipping %s: %s", fpath, exc)
            continue

        bouts, labels = classification_data_to_bouts(
            classification_data,
            C=dcfg.C,
            T=dcfg.T,
            V=mcfg.num_nodes,
            M=dcfg.M,
            fps=dcfg.fps,
            denominator=dcfg.denominator,
            center=dcfg.center_data,
            T2=dcfg.T2,
            align_data=dcfg.align_data,
        )
        if bouts is not None and len(bouts) > 0:
            all_bouts.append(bouts)
            all_labels.append(labels)
            log.info("Loaded %d bouts from %s", len(bouts), fpath.name)

    if not all_bouts:
        raise ValueError(
            "No valid bouts found in the supplied classification files. "
            "Check that the files exist and contain pose data."
        )

    X = np.concatenate(all_bouts, axis=0)   # (N, C, T2, V, M)
    y = np.concatenate(all_labels, axis=0)  # (N,)

    log.info("Total bouts: %d  |  classes: %s", len(X), np.unique(y))

    # Build augmentation flags dict to pass to PoseDataset
    aug_flags = {
        "rotate": aug.rotate,
        "jitter": aug.jitter,
        "scale": aug.scale,
        "shear": aug.shear,
        "roll": aug.roll,
        "fragment": aug.fragment,
    }

    full_dataset = PoseDataset(
        data=X,
        labels=y,
        augmentation=aug_flags,
        num_class=mcfg.num_class,
        T=dcfg.T2,
        C=dcfg.C,
    )

    # Optional class-weight tensor
    class_weight_tensor: Optional[torch.Tensor] = None
    if tcfg.class_weights is not None:
        class_weight_tensor = torch.tensor(tcfg.class_weights, dtype=torch.float32)

    # Train / val split
    val_split = tcfg.val_split
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split)) if val_split > 0 else 0
    n_train = n_total - n_val

    if n_val > 0:
        generator = torch.Generator().manual_seed(tcfg.seed)
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)
    else:
        train_ds = full_dataset
        val_ds = None

    # Weighted sampling for class imbalance
    sampler = None
    if tcfg.class_weights is None:
        # Auto-compute class frequencies
        counts = np.bincount(y[:n_train] if n_val > 0 else y, minlength=mcfg.num_class)
        counts = np.where(counts == 0, 1, counts)
        weights_per_sample = 1.0 / counts[y[:n_train] if n_val > 0 else y]
        sample_weights = torch.from_numpy(weights_per_sample).float()
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader: Optional[DataLoader] = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=tcfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader
