"""Tests for poser.skeletons.registry."""

from __future__ import annotations

import numpy as np

from poser.models.graph import Graph
from poser.skeletons.loader import yaml_to_graph
from poser.skeletons.registry import get_skeleton


def test_openpose_yaml_has_the_18_openpose_joints():
    # The YAML used the 25-keypoint order cut to 18: mid_hip sat at index 8
    # and l_ear was missing, so indices 8 to 17 named the wrong joints.
    assert get_skeleton("openpose").node_names == [
        "nose",
        "neck",
        "r_shoulder",
        "r_elbow",
        "r_wrist",
        "l_shoulder",
        "l_elbow",
        "l_wrist",
        "r_hip",
        "r_knee",
        "r_ankle",
        "l_hip",
        "l_knee",
        "l_ankle",
        "r_eye",
        "l_eye",
        "r_ear",
        "l_ear",
    ]


def test_openpose_yaml_matches_graph_layout():
    from_yaml = yaml_to_graph(get_skeleton("openpose"))

    np.testing.assert_array_equal(from_yaml.A, Graph(layout="openpose").A)
