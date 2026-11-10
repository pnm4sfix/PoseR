"""Segment a recording into bouts: contiguous stretches of one behaviour.

Three strategies, all napari-free:

orthogonal_variance
    Projects movement orthogonally to the current heading. Best for species
    with directed locomotion, such as zebrafish.
egocentric_variance
    Peak-finds on Euclidean movement relative to a centre node. Simpler, and
    not specific to directed locomotion.
manual_bout
    Takes a start and end the user picked by hand.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import scipy.stats as st

BoutList = List[Tuple[int, int]]


EGOCENTRIC_PROMINENCE_FACTOR = 7.0


def _smoothing_sigma(fps: float) -> int:
    """Gaussian width in frames, never below 1.

    gaussian_filter1d divides by sigma squared, so a sigma of 0 raises
    ZeroDivisionError. int(fps / 10) reaches 0 for anything under 10 fps.
    """
    return max(1, int(fps / 10))


def _peak_distance(fps: float) -> int:
    """Minimum frames between bouts, never below 1.

    find_peaks rejects a distance under 1, and int(fps / 2) reaches 0 below
    2 fps.
    """
    return max(1, int(fps / 2))


def check_behaviour_confidence(
    ci: Optional[np.ndarray],
    start: int,
    end: int,
    confidence_threshold: float = 0.8,
) -> bool:
    """Report whether tracking was confident enough across one bout.

    Args:
        ci: Confidence values, shape (V, T) or (T,). None accepts the bout
            without checking.
        start: First frame of the bout. Clamped to 0, because the detectors
            pad bouts outwards and so can start before frame 0.
        end: One past the last frame. Clamped to the length of ci.

    Returns:
        True when the median confidence over the window is at least
        confidence_threshold. False for an empty window, and for one holding
        no usable values.
    """
    if ci is None:
        return True

    first = max(0, start)
    last = min(end, ci.shape[-1])
    if first >= last:
        return False

    window = ci[..., first:last]
    if np.all(np.isnan(window)):
        return False
    return float(np.nanmedian(window)) >= confidence_threshold


def egocentric_variance(
    points: np.ndarray,
    center_node: int,
    fps: float,
    n_nodes: int = 0,
    *,
    confidence_threshold: float = 0.8,
    ci: Optional[np.ndarray] = None,
) -> Tuple[BoutList, np.ndarray, np.ndarray]:
    """Detect locomotion bouts from egocentric Euclidean movement.

    Movement is measured relative to center_node, smoothed, then peak-found.
    Simpler than orthogonal_variance and not specific to directed locomotion.

    Args:
        points: Shape (n_nodes * n_frames, 3), columns (frame, y, x), ordered
            node-major: every frame of node 0, then every frame of node 1.
        center_node: Node the other nodes are measured relative to.
        fps: Frames per second, which sets the smoothing width and the minimum
            spacing between bouts.
        n_nodes: Number of skeleton nodes. Zero infers it from points, which
            is legacy behaviour and often wrong; pass the real value.
        confidence_threshold: Minimum median confidence to keep a bout.
        ci: Confidence values, shape (V, T). None keeps every bout.

    Returns:
        The bouts as (start, end) frame pairs, the smoothed movement signal,
        and the raw per-node Euclidean movement.

    Note:
        Bouts are padded 20 frames either side of the detected peak, so start
        can be negative and end can exceed the recording. Consumers clamp.
        Unlike orthogonal_variance the prominence is not configurable.
    """
    if n_nodes <= 0:
        # Legacy fallback — unreliable if frame count > 1
        n_nodes = int(round(points.shape[0] / (int(points[:, 0].max()) + 1)))
        n_nodes = max(n_nodes, 1)
    reshap = points.reshape(n_nodes, -1, 3)
    reshap = np.nan_to_num(reshap)

    center = reshap[center_node, :, 1:]
    egocentric = reshap.copy()
    egocentric[:, :, 1:] = reshap[:, :, 1:] - center[None, :, :]

    absol_traj = egocentric[:, 1:, 1:] - egocentric[:, :-1, 1:]
    euclidean = np.sqrt(absol_traj[:, :, 0] ** 2 + absol_traj[:, :, 1] ** 2)
    var = np.median(euclidean, axis=0)

    gauss_filtered = gaussian_filter1d(var, _smoothing_sigma(fps))
    amd = np.median(gauss_filtered - gauss_filtered[0]) / 0.6745

    peaks = find_peaks(
        gauss_filtered,
        prominence=amd * EGOCENTRIC_PROMINENCE_FACTOR,
        distance=_peak_distance(fps),
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


def orthogonal_variance(
    points: np.ndarray,
    center_node: int,
    fps: float,
    n_nodes: int,
    *,
    amd_threshold: float = 2.0,
    confidence_threshold: float = 0.8,
    ci: Optional[np.ndarray] = None,
) -> Tuple[BoutList, np.ndarray, float, np.ndarray]:
    """Detect locomotion bouts by projecting movement orthogonally to heading.

    Suits zebrafish and other species with directed locomotion, where turning
    away from the current heading marks the start of a bout.

    Args:
        points: Shape (n_nodes * n_frames, 3), columns (frame, y, x), ordered
            node-major: every frame of node 0, then every frame of node 1.
        center_node: Node the other nodes are measured relative to.
        fps: Frames per second, which sets the smoothing width and the minimum
            spacing between bouts.
        n_nodes: Number of skeleton nodes.
        amd_threshold: Multiplier on the median absolute deviation of the
            smoothed signal, giving the peak prominence. Raise it to detect
            fewer, stronger bouts.
        confidence_threshold: Minimum median confidence to keep a bout.
        ci: Confidence values, shape (V, T). None keeps every bout.

    Returns:
        The bouts as (start, end) frame pairs, the smoothed movement signal,
        the prominence threshold that was used, and the raw per-node Euclidean
        movement.

    Note:
        End can exceed the recording, so consumers clamp. Unlike
        egocentric_variance this pads only at the end, not the start.
    """
    reshap = points.reshape(n_nodes, -1, 3)
    reshap = np.nan_to_num(reshap)

    center = reshap[center_node, :, 1:]
    egocentric = reshap.copy()
    egocentric[:, :, 1:] = reshap[:, :, 1:] - center[None, :, :]

    absol_traj = egocentric[:, 1:, 1:] - egocentric[:, :-1, 1:]
    euclidean = np.sqrt(absol_traj[:, :, 0] ** 2 + absol_traj[:, :, 1] ** 2)

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

    gauss_filtered = gaussian_filter1d(var, _smoothing_sigma(fps))
    amd = st.median_abs_deviation(gauss_filtered)
    threshold = amd * amd_threshold

    peaks = find_peaks(
        gauss_filtered,
        prominence=threshold,
        distance=_peak_distance(fps),
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


def manual_bout(
    start: int,
    end: int,
    coords_data: dict,
    individual_key,
) -> dict:
    """Build a bout from boundaries the user picked by hand.

    Args:
        start: First frame of the bout.
        end: One past the last frame.
        coords_data: Mapping of individual to {"x", "y", "ci"}, each (V, T).
        individual_key: Which individual to slice.

    Returns:
        Keys start, end, coords, ci, classification and bout_method. coords is
        (V * T_window, 3) holding x, y and ci, which is the layout save_to_h5
        writes; ci is repeated there and also given separately as (V * T).

    Note:
        The end frame is under the key "end", matching what the panels read.
        save_to_h5 reads "stop", so a bout from here cannot be saved without
        translating the key first.
    """
    data = coords_data[individual_key]
    # np.array covers both shapes core.io returns: DataFrames from read_dlc,
    # ndarrays from read_sleap and read_poser_coords.
    x = np.array(data["x"])
    y = np.array(data["y"])
    ci = np.array(data["ci"])

    x_win = x[:, start:end]
    y_win = y[:, start:end]
    ci_win = ci[:, start:end]

    coords = np.stack([x_win, y_win, ci_win], axis=-1)  # (V, T_window, 3)
    coords_flat = coords.reshape(-1, 3)
    ci_flat = ci_win.reshape(-1)

    return {
        "start": start,
        "end": end,
        "coords": coords_flat,
        "ci": ci_flat,
        "classification": "",
        "bout_method": "manual",
    }


def _remove_overlaps(bouts: BoutList, gap: int = 10) -> BoutList:
    """Pull overlapping bouts apart, dropping any that collapse to nothing.

    The detectors pad each bout outwards, so neighbours can overlap even when
    the movements they came from did not.
    """
    if len(bouts) < 2:
        return bouts
    b = np.array(bouts)
    overlap = b[1:, 0] - b[:-1, 1]
    overlap_idx = np.where(overlap <= 0)[0] + 1
    b[overlap_idx, 0] = b[overlap_idx, 0] + gap
    b[overlap_idx - 1, 1] = b[overlap_idx, 0] - gap
    return [(int(s), int(e)) for s, e in b.tolist() if e > s]
