"""
Pose augmentation utilities.

All functions operate on a single pose sample with shape (C, T, V, M)
and return an array of shape (num_aug, C, T, V, M).
These were previously duplicated between _loader.py (ZebData methods) and
utils.py (standalone functions).  This is the single authoritative source.
"""

from __future__ import annotations

import numpy as np


def rotate_transform(behaviour: np.ndarray, num_angles: int) -> np.ndarray:
    """Rotate a pose by a random angle in [-30, 30) degrees.

    Training augmentation: each copy gets its own independent angle. Only x and
    y are rotated, the confidence channel is carried through untouched.

    Args:
        behaviour: One pose sample, shape (C, T, V, M).
        num_angles: How many rotated copies to produce.

    Returns:
        Shape (num_angles, C, T, V, M).

    Note:
        Rotation is about the origin, so centre the pose first or it will be
        translated as well. Draws from the global numpy random state, so seed
        np.random to make a run reproducible.
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
    """Add uniform noise in [-2, 2) pixels to the x and y coordinates.

    Training augmentation: every coordinate gets its own independent offset,
    and each copy its own noise field. The confidence channel is carried
    through untouched.

    Args:
        behaviour: One pose sample, shape (C, T, V, M).
        num_jitter: How many jittered copies to produce.

    Returns:
        Shape (num_jitter, C, T, V, M).

    Note:
        Draws from the global numpy random state, so seed np.random to make a
        run reproducible.
    """
    jittered = np.zeros((num_jitter, *behaviour.shape))
    for i in range(num_jitter):
        jitter = (np.random.random(behaviour[:2].shape) * 4) - 2
        jittered[i] = behaviour.copy()
        jittered[i, :2] = behaviour[:2] + jitter
    return jittered


def scale_transform(behaviour: np.ndarray, num_scales: int) -> np.ndarray:
    """Scale the x and y coordinates by a random factor in [0, 3).

    Training augmentation: one factor per copy, applied to every coordinate.
    The confidence channel is carried through untouched.

    Args:
        behaviour: One pose sample, shape (C, T, V, M).
        num_scales: How many scaled copies to produce.

    Returns:
        Shape (num_scales, C, T, V, M).

    Note:
        The range reaches down to zero, so some copies collapse the pose
        towards a point. Draws from the global numpy random state, so seed
        np.random to make a run reproducible.
    """
    scaled = np.zeros((num_scales, *behaviour.shape))
    for i in range(num_scales):
        scale = np.random.random() * 3
        scaled[i] = behaviour.copy()
        scaled[i, :2] = behaviour[:2] * scale
    return scaled


def shear_transform(behaviour: np.ndarray, num_shears: int) -> np.ndarray:
    """Shear the x and y coordinates by random factors.

    Training augmentation: the x factor is drawn from [-1, 1) and the y factor
    from [0, 1), so the shear is asymmetric. The confidence channel is carried
    through untouched.

    Args:
        behaviour: One pose sample, shape (C, T, V, M).
        num_shears: How many sheared copies to produce.

    Returns:
        Shape (num_shears, C, T, V, M).

    Note:
        Draws from the global numpy random state, so seed np.random to make a
        run reproducible.
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
    """Shift the pose sequence in time by a random number of frames.

    Training augmentation: the shift is drawn from [-20, 20] frames and wraps
    around, so frames pushed off one end reappear at the other. All three
    channels move together.

    Args:
        behaviour: One pose sample, shape (C, T, V, M).
        num_rolls: How many rolled copies to produce.

    Returns:
        Shape (num_rolls, C, T, V, M).

    Note:
        Wrapping splices the end of the bout onto its start, so a rolled copy
        is not a contiguous stretch of real movement. Draws from the global
        numpy random state, so seed np.random to make a run reproducible.
    """
    rolled = np.zeros((num_rolls, *behaviour.shape))
    for i in range(num_rolls):
        shift = np.random.randint(-20, 21)
        rolled[i] = np.roll(behaviour, shift, axis=1)
    return rolled


def fragment_transform(
    behaviour: np.ndarray, num_fragments: int
) -> np.ndarray:
    """Cut a random sub-sequence and tile it back to the original length.

    Training augmentation: the start is drawn from [0, T-2] and the length
    from [10, 60] frames, then _pad_to_length repeats the fragment until it
    reaches T again.

    Args:
        behaviour: One pose sample, shape (C, T, V, M).
        num_fragments: How many fragment copies to produce.

    Returns:
        Shape (num_fragments, C, T, V, M).

    Note:
        A fragment starting near the end is shorter than the requested length,
        since the slice is not clamped. Draws from the global numpy random
        state, so seed np.random to make a run reproducible.
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


def random_augmentation(bhv: np.ndarray, num_aug: int = 1) -> np.ndarray:
    """Apply rotate, then jitter, then scale, then shear, once each.

    Args:
        bhv: One pose sample, shape (C, T, V, M).
        num_aug: How many copies each transform generates internally. Only the
            first is kept, so any value produces one output.

    Returns:
        Shape (C, T, V, M), the same as the input.
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
    """Oversample minority classes towards ideal_sample_no.

    Classes already at or above the target are truncated to it exactly.
    Smaller classes are topped up with rotate, jitter, scale, shear and roll
    copies of every sample they hold. Labels below zero are dropped.

    Args:
        data: Pose samples, shape (N, C, T, V, M).
        labels: Integer class labels, shape (N,).
        ideal_sample_no: Target number of samples per class.

    Returns:
        The augmented samples and their labels.

    Note:
        Undersized classes overshoot the target rather than meeting it, because
        every sample contributes a whole set of copies: 10 samples aiming at
        100 yield 110, and 50 yield 150.
    """
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
    values, counts = np.unique(final_labels, return_counts=True)
    print(f"Augmented dataset: {dict(zip(values.tolist(), counts.tolist()))}")
    return final_data, final_labels


def _pad_to_length(pose: np.ndarray, new_t: int) -> np.ndarray:
    """Tile or truncate a (C, T, V, M) array along time to exactly new_t frames.

    An empty time axis returns zeros rather than raising.
    """
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
