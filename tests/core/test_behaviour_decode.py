"""Tests for poser.core.behaviour_decode."""

from __future__ import annotations

import numpy as np

from poser.core.behaviour_decode import _points_from_coords

V = 3
T = 4


def _identity_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Arrays whose values encode node.frame, so row order is readable."""
    y = np.array([[v + f / 10 for f in range(T)] for v in range(V)])
    return y.copy(), y.copy()


def test_rows_are_node_major():
    # orthogonal_variance does points.reshape(n_nodes, -1, 3), so the first T
    # rows must all belong to node 0.
    x, y = _identity_arrays()
    points = _points_from_coords(x, y, V, T)
    node_of_row = [int(value) for value in points[:, 1]]
    assert node_of_row == [0] * T + [1] * T + [2] * T


def test_frame_column_matches_the_coordinates_beside_it():
    # This is the defect the transpose caused: the frame label and the
    # coordinate on the same row came from different frames.
    x, y = _identity_arrays()
    points = _points_from_coords(x, y, V, T)
    for frame_label, y_value in zip(points[:, 0], points[:, 1]):
        assert round((y_value % 1) * 10) == int(frame_label)


def test_shape_and_column_order():
    x, y = _identity_arrays()
    points = _points_from_coords(x, y, V, T)
    assert points.shape == (V * T, 3)
    # columns are (frame, y, x)
    np.testing.assert_allclose(points[:, 1], y.reshape(-1))
    np.testing.assert_allclose(points[:, 2], x.reshape(-1))


def test_one_node_trajectory_is_contiguous_after_reshape():
    x, y = _identity_arrays()
    points = _points_from_coords(x, y, V, T)
    blocks = points.reshape(V, T, 3)
    for node in range(V):
        assert [int(v) for v in blocks[node, :, 1]] == [node] * T
