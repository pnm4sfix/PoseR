"""
Pose augmentation utilities.

All functions operate on a single pose sample with shape (C, T, V, M)
and return an array of shape (num_aug, C, T, V, M).
These were previously duplicated between _loader.py (ZebData methods) and
utils.py (standalone functions).  This is the single authoritative source.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Individual transforms — return (num_aug, C, T, V, M)
# ---------------------------------------------------------------------------

def rotate_transform(behaviour: np.ndarray, num_angles: int) -> np.ndarray:
    """Randomly rotate a pose by ±30 degrees.

    Parameters
    ----------
    behaviour:
        Single pose sample, shape ``(C, T, V, M)``.
    num_angles:
        Number of rotated copies to produce.

    Returns
    -------
    np.ndarray
        Shape ``(num_angles, C, T, V, M)``.
    """
    rotated = np.zeros((num_angles, *behaviour.shape))
    for i in range(num_angles):
        angle = np.radians((np.random.random() * 60) - 30)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, s], [-s, c]])  # clockwise
        transformed = np.dot(R, behaviour[:2].reshape(2, -1)).reshape(
            behaviour[:2].shape
        )
        rotated[i] = behaviour.copy()
        rotated[i, :2] = transformed
    return rotated


def jitter_transform(behaviour: np.ndarray, num_jitter: int) -> np.ndarray:
    """Add small random noise (±2 px) to x/y coordinates.

    Parameters
    ----------
    behaviour:
        Shape ``(C, T, V, M)``.
    num_jitter:
        Number of jittered copies to produce.
    """
    jittered = np.zeros((num_jitter, *behaviour.shape))
    for i in range(num_jitter):
        jitter = (np.random.random(behaviour[:2].shape) * 4) - 2
        jittered[i] = behaviour.copy()
        jittered[i, :2] = behaviour[:2] + jitter
    return jittered


def scale_transform(behaviour: np.ndarray, num_scales: int) -> np.ndarray:
    """Randomly scale pose coordinates by 0–3×.

    Parameters
    ----------
    behaviour:
        Shape ``(C, T, V, M)``.
    num_scales:
        Number of scaled copies to produce.
    """
    scaled = np.zeros((num_scales, *behaviour.shape))
    for i in range(num_scales):
        scale = np.random.random() * 3
        scaled[i] = behaviour.copy()
        scaled[i, :2] = behaviour[:2] * scale
    return scaled


def shear_transform(behaviour: np.ndarray, num_shears: int) -> np.ndarray:
    """Apply random shear transform to pose coordinates.

    Parameters
    ----------
    behaviour:
        Shape ``(C, T, V, M)``.
    num_shears:
        Number of sheared copies to produce.
    """
    sheared = np.zeros((num_shears, *behaviour.shape))
    for i in range(num_shears):
        shear_x = (np.random.random() * 2) - 1
        shear_y = np.random.random()
        shear_matrix = np.array([[1, shear_x], [shear_y, 1]])
        transformed = np.dot(shear_matrix, behaviour[:2].reshape(2, -1)).reshape(
            behaviour[:2].shape
        )
        sheared[i] = behaviour.copy()
        sheared[i, :2] = transformed
    return sheared


def roll_transform(behaviour: np.ndarray, num_rolls: int) -> np.ndarray:
    """Temporally roll (shift) the pose sequence by a random amount.

    Parameters
    ----------
    behaviour:
        Shape ``(C, T, V, M)``.
    num_rolls:
        Number of rolled copies.
    """
    rolled = np.zeros((num_rolls, *behaviour.shape))
    for i in range(num_rolls):
        shift = np.random.randint(-20, 21)
        rolled[i] = np.roll(behaviour, shift, axis=1)
    return rolled


def fragment_transform(
    behaviour: np.ndarray, num_fragments: int
) -> np.ndarray:
    """Extract a random sub-sequence and pad it back to the original length.

    Parameters
    ----------
    behaviour:
        Shape ``(C, T, V, M)``.
    num_fragments:
        Number of fragment copies.
    """
    T = behaviour.shape[1]
    fragments = np.zeros((num_fragments, *behaviour.shape))
    for i in range(num_fragments):
        start = np.random.randint(0, T - 1)
        length = np.random.randint(10, 61)
        fragment = behaviour[:, start : start + length, :, :]
        # pad back to T using simple tiling
        fragments[i] = _pad_to_length(fragment, T)
    return fragments


# ---------------------------------------------------------------------------
# Composite — apply all transforms once (used inline during __getitem__)
# ---------------------------------------------------------------------------

def random_augmentation(bhv: np.ndarray, num_aug: int = 1) -> np.ndarray:
    """Apply rotate → jitter → scale → shear sequentially, once.

    Parameters
    ----------
    bhv:
        Shape ``(C, T, V, M)``.
    num_aug:
        Kept for API compatibility; always produces exactly 1 output (the
        first element of each transform chain).

    Returns
    -------
    np.ndarray
        Shape ``(C, T, V, M)`` — same as input.
    """
    bhv = rotate_transform(bhv, num_aug)[0]
    bhv = jitter_transform(bhv, num_aug)[0]
    bhv = scale_transform(bhv, num_aug)[0]
    bhv = shear_transform(bhv, num_aug)[0]
    return bhv


def dynamic_augmentation(
    data: np.ndarray,
    labels: np.ndarray,
    ideal_sample_no: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes up to *ideal_sample_no* using 5 augmentation types.

    Parameters
    ----------
    data:
        Shape ``(N, C, T, V, M)``.
    labels:
        Shape ``(N,)`` — integer class labels.
    ideal_sample_no:
        Target number of samples per class.

    Returns
    -------
    (augmented_data, augmented_labels)
    """
    import pandas as pd  # local to avoid top-level pandas dep in augmentation

    aug_data: list[np.ndarray] = []
    aug_labels: list[np.ndarray] = []

    for label in np.unique(labels):
        if label < 0:
            continue
        mask = labels == label
        subset = data[mask]
        count = subset.shape[0]

        if count == 0:
            continue

        if count >= ideal_sample_no:
            aug_data.append(subset[:ideal_sample_no])
            aug_labels.append(np.full(ideal_sample_no, label))
            continue

        ratio = ideal_sample_no / count
        n_types = 5
        remainder = int(ratio % n_types)
        n_each = int(ratio / n_types)

        for b in range(count):
            bhv = subset[b].copy()
            pieces = [
                bhv.reshape(1, *bhv.shape),
                rotate_transform(bhv, n_each + remainder),
                jitter_transform(bhv, n_each),
                scale_transform(bhv, n_each),
                shear_transform(bhv, n_each),
                roll_transform(bhv, n_each),
            ]
            combined = np.concatenate(pieces)
            aug_data.append(combined)
            aug_labels.append(np.full(combined.shape[0], label))

    final_data = np.concatenate(aug_data)
    final_labels = np.concatenate(aug_labels)
    print(f"Augmented dataset: {pd.Series(final_labels).value_counts().to_dict()}")
    return final_data, final_labels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad_to_length(pose: np.ndarray, new_t: int) -> np.ndarray:
    """Tile a (C, T, V, M) array along the time axis to reach *new_t* frames."""
    t = pose.shape[1]
    if t == 0:
        return np.zeros((pose.shape[0], new_t, *pose.shape[2:]))
    ratio = new_t / t
    result = pose
    if ratio > 1:
        for _ in range(int(ratio) - 1):
            result = np.concatenate([result, pose], axis=1)
    diff = new_t - result.shape[1]
    if diff > 0:
        result = np.concatenate([result, pose[:, :diff]], axis=1)
    elif diff < 0:
        result = result[:, :new_t]
    return result
