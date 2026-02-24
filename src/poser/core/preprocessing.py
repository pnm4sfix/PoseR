"""
Preprocessing — convert raw coords_data + bout list into model-ready arrays.

This is the napari-free extraction of ``preprocess_bouts()`` and
``classification_data_to_bouts()`` from ``_widget.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .augmentation import _pad_to_length


# ---------------------------------------------------------------------------
# Bout-level transforms (operate on a single (C,T,V,M) bout)
# ---------------------------------------------------------------------------

def center_all(bout: np.ndarray, center_node: int) -> np.ndarray:
    """Subtract centre-node xy from all nodes at every frame."""
    ref = bout[0:2, :, center_node]
    ref = ref.reshape(ref.shape[0], ref.shape[1], 1, ref.shape[2]
                      if ref.ndim == 3 else 1)
    # handle both (...,) and (...,M) shapes
    center_xy = bout[0:2, :, center_node].reshape(
        2, bout.shape[1], 1, -1
    )
    centered = bout.copy()
    centered[0:2] = centered[0:2] - center_xy
    return centered


def align(bout: np.ndarray, head_node: int = 0) -> np.ndarray:
    """Rotate bout so the head node points upward (0,1) at frame 0."""
    nose_vec = bout[0:2, 0, head_node]
    if np.linalg.norm(nose_vec) == 0:
        return bout
    theta = float(np.arctan2(nose_vec[0], nose_vec[1]))
    if theta < 0:
        theta += 2 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    result = bout.copy()
    result[0:2] = np.dot(R, bout[0:2].reshape(2, -1)).reshape(bout[0:2].shape)
    return result


def pad_bout(bout: np.ndarray, new_t: int) -> np.ndarray:
    """Tile a bout in time to reach exactly *new_t* frames (wraps at boundaries)."""
    return _pad_to_length(bout, new_t)


def pad_flip_bout(bout: np.ndarray, new_t: int) -> np.ndarray:
    """Pad by alternating forward/reverse copies (ping-pong)."""
    t = bout.shape[1]
    if t == 0:
        return np.zeros((bout.shape[0], new_t, *bout.shape[2:]))
    result = bout
    flip = np.flip(bout, axis=1)
    fwd = True
    while result.shape[1] < new_t:
        result = np.concatenate([result, flip if not fwd else bout], axis=1)
        fwd = not fwd
    return result[:, :new_t]


def mirror_y(bout: np.ndarray) -> np.ndarray:
    """Flip the y-axis (negate channel 1)."""
    result = bout.copy()
    result[1] *= -1
    return result


# ---------------------------------------------------------------------------
# Main preprocess pipeline
# ---------------------------------------------------------------------------

def preprocess_bouts(
    egocentric: np.ndarray,
    ci: Union[np.ndarray, "pd.DataFrame"],
    bouts: List[Tuple[int, int]],
    *,
    C: int = 3,
    T: Optional[int] = None,
    T2: Optional[int] = None,
    fps: float = 330.0,
    denominator: float = 8.0,
    T_method: str = "window",
    head_node: int = 0,
    return_raw: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert egocentric poses + bout list into ``(N, C, T2, V, M)`` arrays.

    This is the napari-free equivalent of ``PoserWidget.preprocess_bouts()``.

    Parameters
    ----------
    egocentric:
        Shape ``(n_nodes, n_frames, 3)`` — same as ``self.egocentric`` in widget.
    ci:
        Shape ``(n_nodes, n_frames)`` or a DataFrame of same shape.
    bouts:
        List of ``(start, end)`` tuples.
    C, T, T2, fps, denominator, T_method:
        Matching ``config_data`` fields.
    head_node:
        Node index used for ``align()``.
    return_raw:
        If True, also return the un-padded ``(N, C, T, V, M)`` array.

    Returns
    -------
    padded_bouts : ndarray, shape ``(N, C, T2, V, M)``
    dummy_labels : ndarray, shape ``(N,)`` — all zeros (for inference)
    """
    # Resolve T
    if T_method == "window":
        T_effective = 2 * int(fps / denominator)
    elif isinstance(T_method, int):
        T_effective = T_method
    else:
        T_effective = T if T is not None else 43

    # CI array
    if hasattr(ci, "to_numpy"):
        ci_arr = ci.to_numpy()
    else:
        ci_arr = np.array(ci)
    ci_arr = ci_arr.reshape((*ci_arr.shape, 1))

    # Spatial coords only (drop frame index from egocentric)
    points = egocentric[:, :, 1:]          # (n_nodes, n_frames, 2)
    points = np.swapaxes(points, 0, 2)     # (2, n_frames, n_nodes)
    cis = np.swapaxes(ci_arr, 0, 2)        # (1, n_frames, n_nodes, 1)

    N = len(bouts)
    V = points.shape[2]
    M = 1
    T_pad = T2 if T2 is not None else T_effective

    bouts_raw = np.zeros((N, C, T_effective, V, M))
    padded = np.zeros((N, C, T_pad, V, M))
    refined_bouts = list(bouts)

    y_mirror = np.array([[1, 0], [0, -1]])

    for n, (bhv_start, bhv_end) in enumerate(bouts):
        bhv_mid = bhv_start + (bhv_end - bhv_start) / 2
        new_start = int(bhv_mid - T_effective / 2)
        new_end = int(bhv_mid + T_effective / 2)
        new_end = (T_effective - (new_end - new_start)) + new_end

        if T_method == "window":
            refined_bouts[n] = (new_start, new_end)

        bhv = points[:, new_start:new_end]
        ci_slice = cis[:, new_start:new_end]
        ci_slice = ci_slice.reshape((*ci_slice.shape, 1)) if ci_slice.ndim == 3 else ci_slice

        # swap x/y channels (widget convention matches here)
        bhv_rs = bhv.copy()
        bhv_rs[1] = bhv[0]
        bhv_rs[0] = bhv[1]
        bhv = bhv_rs

        # reflect y coordinate
        bhv = bhv.reshape((*bhv.shape, M))
        for frame in range(bhv.shape[1]):
            bhv[:2, frame] = (bhv[:2, frame].T @ y_mirror).T

        # align to heading
        bhv = align(bhv, head_node)

        bouts_raw[n, :2] = bhv
        bouts_raw[n, 2] = ci_slice[0] if ci_slice.ndim >= 3 else 0

        padded[n] = pad_bout(bouts_raw[n], T_pad)

    labels = np.zeros(N)
    if return_raw:
        return padded, labels, bouts_raw, refined_bouts
    return padded, labels


# ---------------------------------------------------------------------------
# classification_data → (N, C, T, V, M) + labels
# ---------------------------------------------------------------------------

def classification_data_to_bouts(
    classification_data: Dict,
    C: int,
    T: Optional[int],
    V: int,
    M: int,
    *,
    fps: float,
    denominator: float,
    center: Optional[int] = None,
    T2: Optional[int] = None,
    align_data: bool = False,
    head_node: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a loaded classification dict to ``(N, C, T2, V, M)`` arrays.

    This merges the logic from both ``utils.classification_data_to_bouts``
    and ``PoserWidget.classification_data_to_bouts`` into one place.

    Parameters
    ----------
    classification_data:
        ``{ind: {bout: {"classification": str, "coords": ndarray, "ci": ndarray}}}``.
    C, T, V, M:
        Data dimensions (channels, time, vertices, individuals).
    fps, denominator:
        Used to compute window size.
    center:
        If not None, centre all bouts on this node.
    T2:
        Final padded length (defaults to T if None).
    align_data:
        Whether to apply heading alignment.
    head_node:
        Node used for alignment.

    Returns
    -------
    (all_bouts, all_labels)
        Shapes ``(N, C, T2, V, M)`` and ``(N,)``.
    """
    from .augmentation import _pad_to_length

    T_effective = 2 * int(fps / denominator) if T is None else T
    T_pad = T2 if T2 is not None else T_effective

    y_mirror = np.array([[1, 0], [0, -1]])

    all_bouts: list[np.ndarray] = []
    all_labels: list = []

    for ind in classification_data.keys():
        behaviour_dict = classification_data[ind]
        N_ind = len(behaviour_dict)
        bout_data = np.zeros((N_ind, C, T_pad, V, M))
        bout_labels: list = []

        for bout_idx, bout_key in enumerate(behaviour_dict.keys()):
            info = behaviour_dict[bout_key]
            coords = info["coords"]           # (n_nodes*T_raw, 3) or (V, T_raw, 3)
            ci_raw = info["ci"]

            # reshape to (V, T_raw, 3)
            coords_rs = coords.reshape(V, -1, 3)
            ci_rs = ci_raw.reshape(V, -1, 1)

            # window from middle
            mid = coords_rs.shape[1] // 2
            half = int(fps / denominator)
            start, end = mid - half, mid + half

            if T is None:
                coords_sub = coords_rs
                ci_sub = ci_rs
            else:
                coords_sub = coords_rs[:, start:end]
                ci_sub = ci_rs[:, start:end]

            # reshape (V, T, 3) → (3, T, V)
            swapped = np.swapaxes(coords_sub, 0, 2)  # (3, T, V)
            new_bout = np.zeros_like(swapped)
            new_bout[0] = swapped[2]   # x
            new_bout[1] = swapped[1]   # y
            swapped_ci = np.swapaxes(ci_sub, 0, 2)
            new_bout[2] = swapped_ci[0]

            new_bout = new_bout.reshape((*new_bout.shape, M))

            # mirror y
            for frame in range(new_bout.shape[1]):
                new_bout[:2, frame] = (new_bout[:2, frame].T @ y_mirror).T

            if center is not None:
                new_bout = center_all(new_bout, center)
            if align_data:
                new_bout = align(new_bout, head_node)

            new_bout = pad_bout(new_bout, T_pad)
            bout_data[bout_idx] = new_bout
            bout_labels.append(info["classification"])

        all_bouts.append(bout_data)
        all_labels.extend(bout_labels)

    return np.concatenate(all_bouts), np.array(all_labels)
