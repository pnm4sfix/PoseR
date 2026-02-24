"""
training/finetune_yolo.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Fine-tune an Ultralytics YOLOv8 pose estimation model on new species data.

This module provides :func:`finetune_yolo`, a thin wrapper around the
Ultralytics Python API that:

1. Accepts a list of annotated image directories (already exported in
   YOLO-pose format via :func:`~poser.core.frame_export.save_frame_as_yolo`).
2. Builds or accepts a ``data.yaml`` describing the dataset.
3. Runs ``model.train(...)`` with sensible defaults that can be overridden.

Usage
-----
::

    from poser.training.finetune_yolo import finetune_yolo

    finetune_yolo(
        images_dir="exported_frames/train",
        val_dir="exported_frames/val",
        base_weights="yolov8m-pose.pt",
        num_keypoints=9,
        epochs=50,
    )
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def finetune_yolo(
    images_dir: str | Path,
    *,
    val_dir: Optional[str | Path] = None,
    base_weights: str | Path = "yolov8m-pose.pt",
    data_yaml: Optional[str | Path] = None,
    num_keypoints: int = 9,
    num_classes: int = 1,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 0.001,
    project: str = "poser_yolo_finetune",
    name: str = "run",
    device: str = "auto",
    freeze_backbone: bool = True,
    extra_kwargs: Optional[dict] = None,
) -> Path:
    """Fine-tune a YOLOv8 pose model.

    Parameters
    ----------
    images_dir:
        Directory containing training images (``images/train/...``) or the
        full YOLO dataset root if ``data_yaml`` is given.
    val_dir:
        Optional validation image directory.  When ``None``, a fraction of
        *images_dir* is used automatically by YOLO.
    base_weights:
        A ``.pt`` model checkpoint to start from (YOLOv8/YOLOv11 pose).
    data_yaml:
        Path to an existing ``data.yaml``.  If ``None``, one is generated
        automatically via :func:`~poser.core.frame_export.build_yolo_data_yaml`.
    num_keypoints:
        Number of body keypoints in the dataset.
    num_classes:
        Number of object classes (usually 1 for single-species tracking).
    epochs:
        Training epochs.
    imgsz:
        Input image size (square).
    batch:
        Batch size.
    lr0:
        Initial learning rate.
    project:
        Output project directory.
    name:
        Run name sub-directory.
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``, or GPU index string.
    freeze_backbone:
        Freeze backbone layers and fine-tune only the head.
    extra_kwargs:
        Any additional keyword arguments forwarded to ``model.train()``.

    Returns
    -------
    run_dir : Path
        Directory containing the training outputs and best weights.

    Raises
    ------
    ImportError
        If ``ultralytics`` is not installed.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLO fine-tuning. "
            "Install it with: pip install ultralytics"
        ) from exc

    images_dir = Path(images_dir)

    # Build data.yaml if not provided
    if data_yaml is None:
        from poser.core.frame_export import build_yolo_data_yaml
        data_yaml = build_yolo_data_yaml(
            images_dir,
            num_keypoints=num_keypoints,
            num_classes=num_classes,
            val_dir=val_dir,
        )
        log.info("Generated data.yaml at %s", data_yaml)

    data_yaml = Path(data_yaml)

    model = YOLO(str(base_weights))

    # Freeze backbone layers (0..9 in YOLOv8)
    if freeze_backbone:
        freeze_layers = list(range(10))
        log.info("Freezing backbone layers 0–9.")
    else:
        freeze_layers = []

    train_kwargs: dict = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        project=project,
        name=name,
        device=device,
        freeze=freeze_layers,
        kpt_shape=[num_keypoints, 3],  # x, y, visibility
        **(extra_kwargs or {}),
    )

    log.info("Starting YOLO fine-tuning: %s", train_kwargs)
    results = model.train(**train_kwargs)

    run_dir = Path(project) / name
    log.info("Fine-tuning complete. Results in %s", run_dir)
    return run_dir
