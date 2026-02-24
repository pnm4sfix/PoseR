"""
Frame export utilities for context-aware finetuning.

Feature 3: "Save current frame" saves the selected frame in the correct
format for the active model type:

* Pose Estimation → YOLO-pose ``.txt`` label + ``.png`` image.
* Behaviour Classification → row appended to ``classification_frames.h5``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Pose estimation → YOLO format
# ---------------------------------------------------------------------------

def save_frame_as_yolo(
    frame_image: np.ndarray,
    keypoints: np.ndarray,
    frame_idx: int,
    source_stem: str,
    output_dir: str,
    *,
    class_id: int = 0,
    conf_threshold: float = 0.0,
) -> Tuple[str, str]:
    """Save one video frame and its keypoints in YOLO-pose format.

    Generates a ``<output_dir>/images/<stem>_<frame>.png`` image file and a
    ``<output_dir>/labels/<stem>_<frame>.txt`` label file compatible with
    ``ultralytics`` YOLO-pose finetuning.

    Parameters
    ----------
    frame_image:
        HWC or CHW uint8 image array for the current frame.
    keypoints:
        Shape ``(n_nodes, 2)`` or ``(n_nodes, 3)`` — x, y (and optional
        confidence).  Coordinates should be in **pixel space**.
    frame_idx:
        Integer frame number (used for file naming).
    source_stem:
        Filename stem of the source video / pose file.
    output_dir:
        Root directory for the YOLO dataset.
    class_id:
        YOLO class id (default 0 = single-class pose).
    conf_threshold:
        Keypoints with confidence below this are set to visibility ``0``.

    Returns
    -------
    (image_path, label_path)
    """
    from PIL import Image  # soft dep — only needed at call time

    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    name = f"{source_stem}_{frame_idx:06d}"
    img_path = os.path.join(img_dir, f"{name}.png")
    lbl_path = os.path.join(lbl_dir, f"{name}.txt")

    # --- Save image ---
    img = frame_image
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    Image.fromarray(img.astype(np.uint8)).save(img_path)

    H, W = img.shape[:2]

    # --- Build bounding box from keypoints ---
    xy = keypoints[:, :2]
    valid = ~np.isnan(xy).any(axis=1)
    if not valid.any():
        return img_path, lbl_path

    x_min, y_min = xy[valid].min(axis=0)
    x_max, y_max = xy[valid].max(axis=0)

    # Add a small margin
    margin_x = max((x_max - x_min) * 0.05, 1.0)
    margin_y = max((y_max - y_min) * 0.05, 1.0)
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(W, x_max + margin_x)
    y_max = min(H, y_max + margin_y)

    # Normalise bbox
    cx = ((x_min + x_max) / 2) / W
    cy = ((y_min + y_max) / 2) / H
    bw = (x_max - x_min) / W
    bh = (y_max - y_min) / H

    # --- Normalise keypoints ---
    has_conf = keypoints.shape[1] == 3
    kp_tokens: list[str] = []
    for i, kp in enumerate(keypoints):
        kx, ky = float(kp[0]) / W, float(kp[1]) / H
        if has_conf:
            vis = 2 if float(kp[2]) >= conf_threshold else 0
        else:
            vis = 2
        if np.isnan(kx) or np.isnan(ky):
            kx, ky, vis = 0.0, 0.0, 0
        kp_tokens.extend([f"{kx:.6f}", f"{ky:.6f}", str(vis)])

    label_line = " ".join(
        [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        + kp_tokens
    )
    with open(lbl_path, "w") as f:
        f.write(label_line + "\n")

    return img_path, lbl_path


def build_yolo_data_yaml(output_dir: str, num_keypoints: int, num_classes: int = 1) -> str:
    """Write a minimal YOLO-pose ``data.yaml`` for ultralytics training.

    Parameters
    ----------
    output_dir:
        Root dir containing ``images/`` and ``labels/``.
    num_keypoints:
        Number of skeleton keypoints.
    num_classes:
        Number of object classes (default 1).

    Returns
    -------
    str
        Path to the written ``data.yaml`` file.
    """
    import yaml

    yaml_path = os.path.join(output_dir, "data.yaml")
    data = {
        "path": os.path.abspath(output_dir),
        "train": "images",
        "val": "images",
        "kpt_shape": [num_keypoints, 3],
        "nc": num_classes,
        "names": {i: f"class{i}" for i in range(num_classes)},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return yaml_path


# ---------------------------------------------------------------------------
# Behaviour classification → classification_frames.h5
# ---------------------------------------------------------------------------

def save_frame_as_classification(
    coords_data: Dict,
    individual_key,
    frame_idx: int,
    label: str,
    source_file: str,
    output_dir: str,
) -> str:
    """Append a single labelled pose frame to ``classification_frames.h5``.

    The saved row contains the full ``(C, n_nodes)`` pose at *frame_idx* and
    can be merged with ``Zebtrain.npy`` during the next training run.

    Parameters
    ----------
    coords_data:
        ``{individual: {"x": array, "y": array, "ci": array}}``.
    individual_key:
        Which individual's data to extract.
    frame_idx:
        Frame to extract.
    label:
        Behaviour label string.
    source_file:
        Original pose file path (for provenance).
    output_dir:
        Directory to write / append ``classification_frames.h5``.

    Returns
    -------
    str
        Path to the h5 file.
    """
    import tables as tb

    class FrameRow(tb.IsDescription):  # type: ignore[misc]
        label = tb.StringCol(64)
        source_file = tb.StringCol(256)
        frame_idx = tb.Int32Col()

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, "classification_frames.h5")
    mode = "a" if os.path.exists(h5_path) else "w"

    data = coords_data[individual_key]

    x = np.array(data["x"])
    y = np.array(data["y"])
    ci = np.array(data["ci"])

    # Handle DataFrames
    if hasattr(x, "values"):
        x = x.values
    if hasattr(y, "values"):
        y = y.values
    if hasattr(ci, "values"):
        ci = ci.values

    n_nodes = x.shape[0]

    # Extract single frame — shape (3, n_nodes)
    x_fr = x[:, frame_idx] if x.ndim == 2 else x[frame_idx]
    y_fr = y[:, frame_idx] if y.ndim == 2 else y[frame_idx]
    ci_fr = ci[:, frame_idx] if ci.ndim == 2 else ci[frame_idx]
    pose = np.stack([x_fr, y_fr, ci_fr], axis=0)  # (3, n_nodes)

    with tb.open_file(h5_path, mode=mode, title="classification_frames") as f:
        # Table for metadata
        if "/frames" not in f:
            table = f.create_table("/", "frames", FrameRow, "Frame metadata")
        else:
            table = f.root.frames

        # Array group
        n_existing = len(f.list_nodes("/", classname="Array")) if mode == "a" else 0
        arr_name = f"frame_{n_existing:06d}"
        f.create_array("/", arr_name, pose, f"Pose at frame {frame_idx}")

        row = table.row
        row["label"] = label
        row["source_file"] = str(source_file)
        row["frame_idx"] = frame_idx
        row.append()
        table.flush()

    print(f"Saved frame {frame_idx} (label={label!r}) to {h5_path}")
    return h5_path


def load_classification_frames(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all saved frames from ``classification_frames.h5``.

    Returns
    -------
    (poses, labels)
        *poses* shape ``(N, 3, n_nodes)``.
        *labels* shape ``(N,)`` — string labels.
    """
    import tables as tb

    poses = []
    labels = []
    with tb.open_file(h5_path, mode="r") as f:
        for arr in f.list_nodes("/", classname="Array"):
            poses.append(arr[:])
        for row in f.root.frames.iterrows():
            label = row["label"]
            labels.append(label.decode("utf-8") if isinstance(label, bytes) else label)

    return np.stack(poses, axis=0), np.array(labels)
