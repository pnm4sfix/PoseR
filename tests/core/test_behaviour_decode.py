"""Tests for poser.core.behaviour_decode."""

from __future__ import annotations

import numpy as np

from poser.core import behaviour_decode
from poser.core.behaviour_decode import _points_from_coords, decode_behaviours
from poser.core.io import save_coords_to_h5
from poser.core.schemas.training import DataConfig, TrainingConfig

V = 3
T = 4
CENTER_NODE = 2


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


def _classifier_input(tmp_path, monkeypatch, xy_ci, center_data: bool) -> np.ndarray:
    """Decode one fixed bout and return the (N, C, T2, V, M) array the model gets."""
    x, y, ci = xy_ci
    # Offset so the animal sits well away from the image origin.
    pose_path = save_coords_to_h5(
        {"ind1": {"x": x + 100, "y": y + 50, "ci": ci}}, tmp_path / "recording"
    )
    seen = {}

    def fake_predict(padded, architecture, checkpoint):
        seen["padded"] = padded
        return np.zeros(len(padded), dtype=int)

    monkeypatch.setattr(
        behaviour_decode, "orthogonal_variance", lambda *args, **kwargs: ([(5, 15)],)
    )
    monkeypatch.setattr(behaviour_decode, "_predict", fake_predict)
    data_cfg = DataConfig(T=10, T2=10, center_node=CENTER_NODE, center_data=center_data)
    decode_behaviours(
        pose_path, "unused.ckpt", TrainingConfig(data=data_cfg), str(tmp_path)
    )
    return seen["padded"]


def test_classifier_input_is_centred_on_the_centre_node(tmp_path, monkeypatch, xy_ci):
    # Absolute image coordinates used to reach the classifier, which is
    # trained on bouts with the centre node at the origin.
    padded = _classifier_input(tmp_path, monkeypatch, xy_ci, center_data=True)
    np.testing.assert_allclose(padded[:, :2, :, CENTER_NODE], 0, atol=1e-9)


def test_center_data_false_leaves_coordinates_absolute(tmp_path, monkeypatch, xy_ci):
    padded = _classifier_input(tmp_path, monkeypatch, xy_ci, center_data=False)
    distance_from_origin = np.linalg.norm(padded[:, :2, :, CENTER_NODE], axis=1)
    assert distance_from_origin.min() > 50


def test_missing_centre_detection_does_not_blank_its_frame(
    tmp_path, monkeypatch, xy_ci
):
    # NaN marks a missing detection. Subtracting a NaN centre would spread it
    # to every node of that frame.
    x, y, ci = xy_ci
    x = x.copy()
    x[CENTER_NODE, 7] = np.nan
    padded = _classifier_input(tmp_path, monkeypatch, (x, y, ci), center_data=True)
    assert not np.isnan(padded).any()
