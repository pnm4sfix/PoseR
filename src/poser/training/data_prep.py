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
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

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

    Accepts both HDF5 classification files (legacy) and numpy pose files
    exported by the Annotation panel (``*_pose.npy`` paired with
    ``*_labels.npy``).  The two formats can be mixed freely in one call.

    **Numpy path** (``*_pose.npy``): each file is treated as a continuous
    recording ``(C, T_total, V, M)``.  A :class:`~poser.core.dataset.PoseDataset`
    is created per file with ``preprocess_frame=True`` so that windowing,
    centering, alignment, and padding all happen lazily inside
    ``__getitem__``.  Frames near video boundaries are reflect-padded rather
    than zero-padded.  A :class:`torch.utils.data.ConcatDataset` is used to
    combine multiple files without ever loading them all into memory.

    **HDF5 legacy path** (``*.h5 / *.hdf5``): bouts are pre-extracted via
    ``classification_data_to_bouts`` and concatenated, then wrapped in a
    single ``PoseDataset``.

    Parameters
    ----------
    classification_files:
        Paths to HDF5 files **or** ``*_pose.npy`` files.  For numpy files the
        matching ``*_labels.npy`` must exist in the same directory.
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

    aug_flags = {
        "rotate": aug.rotate,
        "jitter": aug.jitter,
        "scale": aug.scale,
        "shear": aug.shear,
        "roll": aug.roll,
        "fragment": aug.fragment,
    }

    numpy_datasets: List[PoseDataset] = []    # one per *_pose.npy file
    hdf5_bouts:    List[np.ndarray]   = []
    hdf5_labels:   List[np.ndarray]   = []

    for fpath in classification_files:
        fpath = Path(fpath)

        # ── numpy continuous-recording format (*_pose.npy) ───────────
        if fpath.name.endswith("_pose.npy"):
            labels_path = fpath.parent / (fpath.stem[: -len("_pose")] + "_labels.npy")
            if not labels_path.exists():
                log.warning("No labels file for %s — skipping.", fpath)
                continue
            try:
                pose   = np.load(fpath)        # (C, T_total, V, M)
                lbl    = np.load(labels_path)  # (T_total,) int
            except Exception as exc:
                log.warning("Skipping %s: %s", fpath, exc)
                continue

            ds = PoseDataset(
                data=pose,
                labels=lbl.astype(np.int64),
                preprocess_frame=True,
                window_size=dcfg.T,
                transform=dcfg.transforms,
                T=dcfg.T2,
                augmentation=aug_flags,
                center_node=dcfg.center_node,
                head_node=dcfg.head_node,
                num_class=mcfg.num_class,
                C=dcfg.C,
            )
            numpy_datasets.append(ds)
            log.info("Registered numpy recording %s — %d frames.", fpath.name, len(lbl))
            continue

        # ── legacy HDF5 format ────────────────────────────────────────
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
            hdf5_bouts.append(bouts)
            hdf5_labels.append(labels)
            log.info("Loaded %d bouts from %s", len(bouts), fpath.name)

    # ── Assemble full dataset ─────────────────────────────────────────
    all_datasets: List = list(numpy_datasets)

    if hdf5_bouts:
        X = np.concatenate(hdf5_bouts, axis=0)   # (N, C, T2, V, M)
        y = np.concatenate(hdf5_labels, axis=0)  # (N,)
        log.info("HDF5 bouts: %d  |  classes: %s", len(X), np.unique(y))
        hdf5_ds = PoseDataset(
            data=X,
            labels=y,
            augmentation=aug_flags,
            num_class=mcfg.num_class,
            T=dcfg.T2,
            C=dcfg.C,
        )
        all_datasets.append(hdf5_ds)

    if not all_datasets:
        raise ValueError(
            "No valid data found in the supplied files. "
            "Check that files exist and that *_pose.npy files have matching *_labels.npy."
        )

    full_dataset = ConcatDataset(all_datasets) if len(all_datasets) > 1 else all_datasets[0]

    # ── Auto-detect num_nodes and in_channels from data shape ─────────
    # When layout='auto' (default), the model needs to know V and C before
    # it can build the fully-connected graph.  Read them from the first dataset
    # and write them back into the config so from_training_config gets the
    # right values.
    if mcfg.layout == "auto" and numpy_datasets:
        ds0 = numpy_datasets[0]
        if hasattr(ds0, "V") and ds0.V is not None:
            log.info(
                "Auto layout: setting num_nodes=%d from data shape (was %d).",
                ds0.V, mcfg.num_nodes,
            )
            mcfg.num_nodes = ds0.V
        if hasattr(ds0, "C") and ds0.C is not None:
            log.info(
                "Auto layout: setting in_channels=%d from data shape (was %d).",
                ds0.C, mcfg.in_channels,
            )
            mcfg.in_channels = ds0.C
    elif mcfg.layout == "auto" and hdf5_bouts:
        # HDF5 path: shape is (N, C, T, V, M)
        ds0 = all_datasets[-1]  # hdf5_ds was appended last
        if hasattr(ds0, "data") and ds0.data is not None and ds0.data.ndim == 5:
            _, C_det, _, V_det, _ = ds0.data.shape
            log.info("Auto layout: setting num_nodes=%d, in_channels=%d from HDF5 data.", V_det, C_det)
            mcfg.num_nodes = V_det
            mcfg.in_channels = C_det

    # Collect all labels for class weighting (needed before split)
    all_labels_list: List[np.ndarray] = []
    for ds in all_datasets:
        if hasattr(ds, "labels"):
            all_labels_list.append(ds.labels)
    y_all = np.concatenate(all_labels_list, axis=0) if all_labels_list else np.array([], dtype=np.int64)

    # ── Train / val split ─────────────────────────────────────────────
    val_split = tcfg.val_split
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_split)) if val_split > 0 else 0
    n_train = n_total - n_val

    if n_train < 1:
        # Dataset too small to split (e.g. single-file smoke-test).
        # Reuse the full dataset for both train and val so nothing crashes.
        log.warning(
            "Dataset too small (%d samples) to split at val_split=%.2f — "
            "reusing full dataset for both train and val.",
            n_total, val_split,
        )
        train_ds = full_dataset
        val_ds   = full_dataset
        n_train  = n_total
    elif n_val > 0:
        generator = torch.Generator().manual_seed(tcfg.seed)
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)
    else:
        train_ds, val_ds = full_dataset, None

    # ── Weighted sampler (class-balance, enabled by default) ────────────
    # ideal_sample_no inflates the virtual epoch — rare classes get
    # proportionally more draws, matching the legacy ZebData behaviour.
    sampler = None
    if tcfg.weighted_random_sampler and len(y_all) > 0 and tcfg.class_weights is None:
        train_idx = (
            train_ds.indices
            if isinstance(train_ds, Subset)
            else np.arange(len(train_ds))
        )
        # Use modulo so the full-dataset-reuse path (train_ds == full_dataset)
        # never overflows y_all when n_train == n_total.
        y_train = y_all[np.asarray(train_idx) % len(y_all)]
        # Give unlabelled frames (-1) zero sampling weight — they are ignored
        # by the loss function and should not distort class-balance sampling.
        labelled_mask = y_train >= 0
        counts = np.bincount(y_train[labelled_mask].clip(0), minlength=mcfg.num_class)
        counts = np.where(counts == 0, 1, counts)   # avoid 1/0 for absent classes
        w_per_sample = np.where(
            labelled_mask,
            1.0 / counts[y_train.clip(0)],
            0.0,
        )
        sample_weights = torch.from_numpy(w_per_sample).float()
        num_samples = dcfg.ideal_sample_no if dcfg.ideal_sample_no is not None else len(train_ds)
        log.info("WeightedRandomSampler: num_samples=%d (ideal_sample_no=%s)",
                 num_samples, dcfg.ideal_sample_no)
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
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
