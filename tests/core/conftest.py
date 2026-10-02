"""Fixtures for the core layer tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tables as tb

V = 5
T = 20
SEED = 20260916


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def xy_ci(rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic x, y and confidence arrays, each (V, T)."""
    x = rng.normal(0, 1, (V, T))
    y = rng.normal(0, 1, (V, T))
    ci = np.full((V, T), 0.9)
    return x, y, ci


@pytest.fixture
def dlc_frame(xy_ci) -> pd.DataFrame:
    """Single-animal DeepLabCut frame, (T, V * 3) with a 3-level column index."""
    x, y, ci = xy_ci
    columns = pd.MultiIndex.from_product(
        [["scorer1"], [f"node{v}" for v in range(V)], ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    values = np.empty((T, V * 3))
    for v in range(V):
        values[:, v * 3 + 0] = x[v]
        values[:, v * 3 + 1] = y[v]
        values[:, v * 3 + 2] = ci[v]
    return pd.DataFrame(values, columns=columns)


@pytest.fixture
def dlc_h5(tmp_path, dlc_frame) -> str:
    path = tmp_path / "recording_dlc.h5"
    dlc_frame.to_hdf(path, key="df_with_missing", mode="w")
    return str(path)


@pytest.fixture
def dlc_csv(tmp_path, dlc_frame) -> str:
    path = tmp_path / "recording_dlc.csv"
    dlc_frame.to_csv(path)
    return str(path)


@pytest.fixture
def sleap_h5(tmp_path, xy_ci) -> str:
    """SLEAP layout: one group per individual holding x, y and ci."""
    x, y, ci = xy_ci
    path = tmp_path / "recording_sleap.h5"
    with tb.open_file(str(path), mode="w") as f:
        group = f.create_group("/", "track0")
        f.create_array(group, "x", x)
        f.create_array(group, "y", y)
        f.create_array(group, "ci", ci)
    return str(path)


@pytest.fixture
def behaviour_schema():
    """PyTables label-table schema, mirroring _widget.Behaviour."""

    class Behaviour(tb.IsDescription):
        number = tb.Int32Col()
        classification = tb.StringCol(16)
        n_nodes = tb.Int32Col()
        start = tb.Int32Col()
        stop = tb.Int32Col()

    return Behaviour


@pytest.fixture
def classification_data(rng) -> dict:
    """One individual with two labelled bouts, in save_to_h5 input layout."""
    n = V * 10
    return {
        1: {
            behaviour: {
                "classification": label,
                "coords": rng.normal(0, 1, (n, 3)),
                "ci": np.full(n, 0.9),
                "start": behaviour * 25,
                "stop": behaviour * 25 + 10,
            }
            for behaviour, label in enumerate(["swim", "turn"])
        }
    }
