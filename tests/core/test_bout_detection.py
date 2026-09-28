"""Tests for poser.core.bout_detection."""

from __future__ import annotations

import numpy as np
import pytest

from poser.core.bout_detection import check_behaviour_confidence

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
