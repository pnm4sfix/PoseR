"""Tests for poser.core.bout_detection."""

from __future__ import annotations

import numpy as np
import pytest

from poser.core.bout_detection import (
    check_behaviour_confidence,
    egocentric_variance,
    orthogonal_variance,
)

V = 5
T = 400


@pytest.fixture
def confident() -> np.ndarray:
    """(V, T) confidence array, high everywhere."""
    return np.full((V, T), 0.95)


def test_confident_interior_bout_is_accepted(confident):
    assert check_behaviour_confidence(confident, 100, 140)


def test_low_confidence_bout_is_rejected():
    assert not check_behaviour_confidence(np.full((V, T), 0.3), 100, 140)


def test_bout_starting_before_frame_zero_is_still_checked(confident):
    # The detectors pad bouts outwards, so start can be negative. A negative
    # slice index used to read from the far end of the array and return an
    # empty window, rejecting a perfectly confident bout.
    assert check_behaviour_confidence(confident, -15, 25)


def test_bout_running_past_the_end_is_still_checked(confident):
    assert check_behaviour_confidence(confident, T - 10, T + 50)


def test_window_entirely_outside_the_recording_is_rejected(confident):
    assert not check_behaviour_confidence(confident, T + 100, T + 200)


def test_empty_window_is_rejected(confident):
    assert not check_behaviour_confidence(confident, 10, 10)


def test_none_confidence_accepts_the_bout():
    assert check_behaviour_confidence(None, 0, 10)


def test_one_dimensional_confidence_is_accepted():
    assert check_behaviour_confidence(np.full(T, 0.95), -5, 20)


def test_all_nan_window_is_rejected_without_warning(confident):
    with np.errstate(all="raise"):
        assert not check_behaviour_confidence(np.full((V, T), np.nan), 100, 140)


def test_threshold_is_inclusive(confident):
    exact = np.full((V, T), 0.8)
    assert check_behaviour_confidence(exact, 100, 140, confidence_threshold=0.8)
    assert not check_behaviour_confidence(exact, 100, 140, confidence_threshold=0.81)


# --- detectors ---------------------------------------------------------------


@pytest.fixture
def moving_points() -> np.ndarray:
    """(V*T, 3) points with body-relative motion and realistic jitter.

    The jitter is not decoration. orthogonal_variance derives its peak
    prominence from the median absolute deviation of the smoothed signal, and
    a perfectly flat baseline has a MAD of zero, which collapses the threshold
    and makes every result meaningless.
    """
    rng = np.random.default_rng(0)
    t = np.arange(T)
    x = np.zeros((V, T))
    y = np.zeros((V, T))
    x[0] = t * 0.05
    y[0] = np.sin(t / 50.0)
    for v in range(1, V):
        x[v] = x[0] - v * 2.0
        y[v] = y[0]
    for start in range(60, T - 60, 80):
        beat = np.sin(np.linspace(0, 4 * np.pi, 20)) * rng.uniform(2.0, 6.0)
        for v in range(1, V):
            y[v, start : start + 20] += beat * (v / V)
    x += rng.normal(0, 0.15, (V, T))
    y += rng.normal(0, 0.15, (V, T))
    return np.column_stack(
        [np.tile(np.arange(T, dtype=float), V), y.reshape(-1), x.reshape(-1)]
    )


def test_egocentric_variance_no_longer_takes_amd_threshold(moving_points):
    # It accepted the argument and then ignored it, hardcoding the prominence.
    with pytest.raises(TypeError):
        egocentric_variance(moving_points, 0, 25, V, amd_threshold=2.0)


@pytest.mark.parametrize("fps", [60, 25, 9, 2, 1])
def test_detectors_survive_low_frame_rates(moving_points, fps):
    # int(fps / 10) hits 0 below 10 fps, and gaussian_filter1d divides by
    # sigma squared, so this used to raise ZeroDivisionError.
    egocentric_variance(moving_points, 0, fps, V)
    orthogonal_variance(moving_points, 0, fps, V)


def test_raising_amd_threshold_detects_fewer_bouts(moving_points):
    counts = [
        len(orthogonal_variance(moving_points, 0, 25, V, amd_threshold=a)[0])
        for a in (0.5, 2.0, 8.0, 30.0)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_orthogonal_variance_returns_the_threshold_it_used(moving_points):
    _, _, threshold, _ = orthogonal_variance(
        moving_points, 0, 25, V, amd_threshold=8.0
    )
    assert threshold > 0
