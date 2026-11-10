"""Tests for poser.core.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poser.core.preprocessing import preprocess_bouts

V = 5
T = 400


@pytest.fixture
def egocentric() -> np.ndarray:
    """(V, T, 3) array laid out as preprocess_bouts expects: frame, y, x."""
    t = np.arange(T)
    x = np.stack([np.sin(t / 8.0) * 3 + v for v in range(V)])
    y = np.stack([np.cos(t / 8.0) * 3 + v for v in range(V)])
    out = np.zeros((V, T, 3))
    out[:, :, 0] = t
    out[:, :, 1] = y
    out[:, :, 2] = x
    return out


@pytest.fixture
def ci_frame() -> pd.DataFrame:
    return pd.DataFrame(np.full((V, T), 0.95))


def test_window_is_clamped_at_the_start_of_the_recording(egocentric, ci_frame):
    # A bout closer to frame 0 than half a window cannot be centred. Before
    # clamping this sliced from the far end of the array and came back empty.
    padded, _ = preprocess_bouts(
        egocentric, ci_frame, [(5, 10)], C=3, T=100, T2=100, T_method="pad"
    )
    assert padded.shape == (1, 3, 100, V, 1)
    assert not np.isnan(padded).any()


def test_window_is_clamped_at_the_end_of_the_recording(egocentric, ci_frame):
    padded, _ = preprocess_bouts(
        egocentric, ci_frame, [(T - 6, T - 1)], C=3, T=100, T2=100, T_method="pad"
    )
    assert padded.shape == (1, 3, 100, V, 1)
    assert not np.isnan(padded).any()


def test_every_bout_survives_clamping(egocentric, ci_frame):
    bouts = [(5, 10), (200, 210), (T - 6, T - 1)]
    padded, labels = preprocess_bouts(
        egocentric, ci_frame, bouts, C=3, T=100, T2=100, T_method="pad"
    )
    assert padded.shape[0] == len(bouts)
    assert labels.shape == (len(bouts),)


def test_interior_bouts_are_unaffected_by_the_clamp(egocentric, ci_frame):
    # A bout with room either side must be untouched by the bounds check.
    padded, _ = preprocess_bouts(
        egocentric, ci_frame, [(200, 210)], C=3, T=100, T2=100, T_method="pad"
    )
    assert padded.shape == (1, 3, 100, V, 1)
    assert not np.isnan(padded).any()


def test_window_method_still_derives_T_from_fps(egocentric, ci_frame):
    # T_method="window" computes T as 2*int(fps/denominator), which is the
    # legacy path the batch pipeline used before it read the config.
    padded, _ = preprocess_bouts(
        egocentric, ci_frame, [(200, 210)], fps=30.0, denominator=8, T2=50
    )
    assert padded.shape == (1, 3, 50, V, 1)
