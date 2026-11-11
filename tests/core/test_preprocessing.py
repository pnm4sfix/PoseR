"""Tests for poser.core.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poser.core.preprocessing import (
    align,
    classification_data_to_bouts,
    preprocess_bouts,
)

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


def test_align_rotates_each_individual_by_its_own_heading():
    # (C, T, V, M) with two individuals whose heads point right and down.
    # The two angles used to be forced into one float, which raised TypeError
    # for any bout with more than one individual.
    bout = np.zeros((3, 4, 2, 2))
    bout[0:2, :, 0, 0] = [[2.0], [0.0]]
    bout[0:2, :, 0, 1] = [[0.0], [-3.0]]
    bout[2] = 0.9

    aligned = align(bout, head_node=0)

    np.testing.assert_allclose(aligned[0:2, 0, 0, 0], [0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(aligned[0:2, 0, 0, 1], [0.0, 3.0], atol=1e-12)
    np.testing.assert_array_equal(aligned[2], bout[2])
    for m in range(2):
        np.testing.assert_allclose(aligned[..., m : m + 1], align(bout[..., m : m + 1]))


def test_recording_shorter_than_the_window_is_tiled_to_length(egocentric, ci_frame):
    # 60 frames cannot fill a 100-frame window. Assigning the short slice to
    # the fixed-size raw array used to raise a broadcasting ValueError.
    short = egocentric[:, :60]
    padded, _ = preprocess_bouts(
        short, ci_frame.iloc[:, :60], [(5, 10)], C=3, T=100, T2=100, T_method="pad"
    )
    assert padded.shape == (1, 3, 100, V, 1)
    # Frames past the end of the recording repeat it from the start.
    np.testing.assert_array_equal(padded[0, :, 60:], padded[0, :, :40])
    np.testing.assert_array_equal(padded[0, 2], 0.95)


def test_confidence_shorter_than_the_coordinates_is_rejected(egocentric, ci_frame):
    # Tiling is for a recording that is short. A confidence array that stops
    # before the bout is a mismatch between the two inputs, not a short clip.
    with pytest.raises(ValueError, match="ci has 150 frames"):
        preprocess_bouts(
            egocentric,
            ci_frame.iloc[:, :150],
            [(300, 310)],
            C=3,
            T=100,
            T2=100,
            T_method="pad",
        )


def test_xy_only_output_has_no_confidence_channel(egocentric, ci_frame):
    # DataConfig allows C=2. Writing confidence into channel 2 of a
    # two-channel array raised IndexError.
    bouts = [(200, 210)]
    xy, _ = preprocess_bouts(
        egocentric, ci_frame, bouts, C=2, T=100, T2=100, T_method="pad"
    )
    xyc, _ = preprocess_bouts(
        egocentric, ci_frame, bouts, C=3, T=100, T2=100, T_method="pad"
    )
    assert xy.shape == (1, 2, 100, V, 1)
    np.testing.assert_array_equal(xy, xyc[:, :2])


def _classification_data(egocentric: np.ndarray, n_frames: int) -> dict:
    """One labelled bout of n_frames, in the layout read_classification_h5 returns."""
    return {
        1: {
            1: {
                "classification": "left",
                "coords": egocentric[:, :n_frames].reshape(-1, 3),
                "ci": np.full(V * n_frames, 0.95),
            }
        }
    }


def test_classification_crop_is_T_frames_from_the_middle(egocentric):
    # DataConfig defaults. The crop used to be int(fps/denominator) frames
    # either side of the middle, which is 0 here, so every bout came back empty.
    bouts, labels = classification_data_to_bouts(
        _classification_data(egocentric, 300),
        C=3, T=100, V=V, M=1, fps=25, denominator=100.0, T2=100,
    )
    assert bouts.shape == (1, 3, 100, V, 1)
    np.testing.assert_allclose(bouts[0, 0, :, :, 0], egocentric[:, 100:200, 2].T)
    assert list(labels) == ["left"]


def test_classification_crop_is_clamped_to_a_short_bout(egocentric):
    # A bout shorter than T gave a negative start, which sliced the last
    # frames of the bout instead of all of it.
    bouts, _ = classification_data_to_bouts(
        _classification_data(egocentric, 60),
        C=3, T=100, V=V, M=1, fps=400, denominator=8, T2=100,
    )
    np.testing.assert_allclose(bouts[0, 0, :60, :, 0], egocentric[:, :60, 2].T)


def test_classification_bouts_keep_only_the_requested_channels(egocentric):
    # C=2 means xy without confidence. The bout is always built with three
    # channels, so storing it whole into a two-channel array used to raise.
    data = _classification_data(egocentric, 20)
    kwargs = dict(T=10, V=V, M=1, fps=25, denominator=5.0, T2=16)

    xy, labels = classification_data_to_bouts(data, C=2, **kwargs)
    xyc, _ = classification_data_to_bouts(data, C=3, **kwargs)

    assert xy.shape == (1, 2, 16, V, 1)
    np.testing.assert_array_equal(xy, xyc[:, :2])
    assert list(labels) == ["left"]
