"""
Locomotion / bout detection — pure numpy functions, no napari dependency.

Three strategies are available:

* ``orthogonal_variance``  — orthogonal-projection peak finding (best for fish)
* ``egocentric_variance``  — Euclidean median peak finding (simpler)
* ``manual_bout``          — user-defined start/stop pair (all species)
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import scipy.stats as st

BoutList = List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Confidence helper
# ---------------------------------------------------------------------------

def check_behaviour_confidence(
    ci: np.ndarray,
    start: int,
    end: int,
    confidence_threshold: float = 0.8,
) -> bool:
    """Return True if the median confidence in [start, end] is above threshold.

    Parameters
    ----------
    ci:
        Confidence array, shape ``(n_nodes, n_frames)`` or ``(n_frames,)``.
    start, end:
        Frame indices.
    confidence_threshold:
        Minimum required median confidence.
    """
    if ci is None:
        return True
    end_clipped = min(end, ci.shape[-1])
    if start >= end_clipped:
        return False
    window = ci[..., start:end_clipped]
    return float(np.nanmedian(window)) >= confidence_threshold


# ---------------------------------------------------------------------------
# Egocentric (Euclidean median)
# ---------------------------------------------------------------------------

def egocentric_variance(
    points: np.ndarray,
    center_node: int,
    fps: float,
    *,
    amd_threshold: float = 2.0,
    confidence_threshold: float = 0.8,
    ci: np.ndarray = None,
) -> Tuple[BoutList, np.ndarray, np.ndarray]:
    """Detect locomotion bouts using egocentric Euclidean variance.

    Parameters
    ----------
    points:
        Shape ``(n_frames * n_nodes, 3)`` — same layout as ``self.points`` in
        the widget (frame, y, x interleaved for each node).
    center_node:
        Index of the reference/center node.
    fps:
        Frames per second.
    amd_threshold:
        Prominence multiplier relative to the MAD of the smoothed signal.
    confidence_threshold:
        Minimum median CI to accept a bout.
    ci:
        Optional confidence array, shape ``(n_nodes, n_frames)``.

    Returns
    -------
    (bouts, gauss_filtered, euclidean)
        *bouts* is a list of ``(start, end)`` tuples.
    """
    n_nodes = int(points[:, 0].max()) + 1 if points.ndim == 2 else points.shape[0]
    reshap = points.reshape(n_nodes, -1, 3)
    reshap = np.nan_to_num(reshap)

    center = reshap[center_node, :, 1:]
    egocentric = reshap.copy()
    egocentric[:, :, 1:] = reshap[:, :, 1:] - center[None, :, :]

    absol_traj = egocentric[:, 1:, 1:] - egocentric[:, :-1, 1:]
    euclidean = np.sqrt(np.abs(absol_traj[:, :, 0] ** 2 + absol_traj[:, :, 1] ** 2))
    var = np.median(euclidean, axis=0)

    gauss_filtered = gaussian_filter1d(var, int(fps / 10))
    amd = np.median(gauss_filtered - gauss_filtered[0]) / 0.6745

    peaks = find_peaks(
        gauss_filtered,
        prominence=amd * 7,
        distance=int(fps / 2),
        width=5,
        rel_height=0.6,
    )

    bouts: BoutList = [
        (int(start) - 20, int(end) + 20)
        for start, end in zip(peaks[1]["left_ips"], peaks[1]["right_ips"])
        if end > start
    ]

    if ci is not None:
        bouts = [
            (s, e)
            for s, e in bouts
            if check_behaviour_confidence(ci, s, e, confidence_threshold)
        ]

    bouts = _remove_overlaps(bouts)
    return bouts, gauss_filtered, euclidean


# ---------------------------------------------------------------------------
# Orthogonal projection
# ---------------------------------------------------------------------------

def orthogonal_variance(
    points: np.ndarray,
    center_node: int,
    fps: float,
    n_nodes: int,
    *,
    amd_threshold: float = 2.0,
    confidence_threshold: float = 0.8,
    ci: np.ndarray = None,
) -> Tuple[BoutList, np.ndarray, float, np.ndarray]:
    """Detect locomotion bouts using orthogonal-projection peak detection.

    Best suited for zebrafish and other species with directed locomotion.

    Parameters
    ----------
    points:
        Shape ``(n_frames * n_nodes, 3)`` (frame index, y, x).
    center_node:
        Index of reference node.
    fps:
        Frames per second.
    n_nodes:
        Total number of skeleton nodes.
    amd_threshold:
        Threshold multiplier on the MAD-based noise estimate.
    confidence_threshold:
        Minimum median CI to accept a bout.
    ci:
        Optional confidence array, shape ``(n_nodes, n_frames)``.

    Returns
    -------
    (bouts, gauss_filtered, threshold, euclidean)
    """
    reshap = points.reshape(n_nodes, -1, 3)
    reshap = np.nan_to_num(reshap)

    center = reshap[center_node, :, 1:]
    egocentric = reshap.copy()
    egocentric[:, :, 1:] = reshap[:, :, 1:] - center[None, :, :]

    absol_traj = egocentric[:, 1:, 1:] - egocentric[:, :-1, 1:]
    euclidean = np.sqrt(
        np.abs(absol_traj[:, :, 0] ** 2 + absol_traj[:, :, 1] ** 2)
    )

    projections = []
    for n in range(n_nodes):
        traj = absol_traj[n]
        orth = np.flip(traj, axis=1).copy()
        orth[:, 0] = -orth[:, 0]

        future = traj[1:]
        present_orth = orth[:-1]
        denom = np.linalg.norm(present_orth, axis=1)
        denom[denom == 0] = 1

        proj = np.abs(np.sum(future * present_orth, axis=1) / denom)
        proj[np.isnan(proj)] = 0
        projections.append(proj)

    proj_arr = np.array(projections)
    var = np.median(proj_arr, axis=0)

    gauss_filtered = gaussian_filter1d(var, int(fps / 10))
    amd = st.median_abs_deviation(gauss_filtered)
    threshold = amd * amd_threshold

    peaks = find_peaks(
        gauss_filtered,
        prominence=threshold,
        distance=int(fps / 2),
        width=5,
        rel_height=0.6,
    )

    bouts: BoutList = [
        (int(start), int(end))
        for start, end in zip(peaks[1]["left_ips"], peaks[1]["right_ips"])
        if end > start
    ]

    if ci is not None:
        bouts = [
            (s, e)
            for s, e in bouts
            if check_behaviour_confidence(ci, s, e, confidence_threshold)
        ]

    bouts = _remove_overlaps(bouts)
    return bouts, gauss_filtered, threshold, euclidean


# ---------------------------------------------------------------------------
# Manual bout
# ---------------------------------------------------------------------------

def manual_bout(
    start: int,
    end: int,
    coords_data: dict,
    individual_key,
) -> dict:
    """Create a bout dict from a user-defined start/stop pair.

    Parameters
    ----------
    start, end:
        Frame indices.
    coords_data:
        ``{individual: {"x": array, "y": array, "ci": array}}``.
    individual_key:
        Which individual to extract from ``coords_data``.

    Returns
    -------
    dict
        ``{"start": int, "stop": int, "coords": ndarray, "ci": ndarray,
           "classification": "", "bout_method": "manual"}``
    """
    data = coords_data[individual_key]
    x = np.array(data["x"])
    y = np.array(data["y"])
    ci = np.array(data["ci"])

    # Handle DataFrame vs ndarray
    if hasattr(x, "values"):
        x = x.values
    if hasattr(y, "values"):
        y = y.values
    if hasattr(ci, "values"):
        ci = ci.values

    # Slice window: shape (n_nodes, n_frames_in_window)
    x_win = x[:, start:end]
    y_win = y[:, start:end]
    ci_win = ci[:, start:end]

    # Stack to (T_window, n_nodes, 3) for storage — same as classification h5
    n_nodes, T = x_win.shape
    coords = np.stack([x_win, y_win, ci_win], axis=-1)  # (n_nodes, T, 3)
    coords_flat = coords.reshape(-1, 3)  # (n_nodes * T, 3) — matches save_to_h5 format
    ci_flat = ci_win.reshape(-1)

    return {
        "start": start,
        "stop": end,
        "coords": coords_flat,
        "ci": ci_flat,
        "classification": "",
        "bout_method": "manual",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_overlaps(bouts: BoutList, gap: int = 10) -> BoutList:
    """Resolve overlapping bout boundaries by shrinking them."""
    if len(bouts) < 2:
        return bouts
    b = np.array(bouts)
    overlap = b[1:, 0] - b[:-1, 1]
    overlap_idx = np.where(overlap <= 0)[0] + 1
    b[overlap_idx, 0] = b[overlap_idx, 0] + gap
    b[overlap_idx - 1, 1] = b[overlap_idx, 0] - gap
    return [(int(s), int(e)) for s, e in b.tolist() if e > s]
