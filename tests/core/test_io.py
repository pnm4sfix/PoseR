"""Tests for poser.core.io."""

from __future__ import annotations

import numpy as np
import pytest

from poser.core.exceptions import (
    BehaviourWriteError,
    PoseFormatError,
    PoseRError,
    UnsupportedFormatError,
)
from poser.core.io import (
    convert_dlc_to_ctvm,
    read_classification_h5,
    read_coords,
    read_dlc,
    read_poser_coords,
    read_sleap,
    save_coords_to_h5,
    save_to_h5,
)

from .conftest import T, V

COORD_KEYS = {"x", "y", "ci"}


# --- readers ---------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["dlc_h5", "dlc_csv"])
def test_read_dlc_handles_both_extensions(request, fixture_name):
    coords = read_dlc(request.getfixturevalue(fixture_name))
    assert list(coords) == ["individual1"]
    assert set(coords["individual1"]) == COORD_KEYS
    assert coords["individual1"]["x"].shape == (V, T)


def test_read_dlc_returns_dataframes(dlc_h5):
    import pandas as pd

    coords = read_dlc(dlc_h5)["individual1"]
    assert all(isinstance(coords[key], pd.DataFrame) for key in COORD_KEYS)


def test_read_dlc_preserves_values(dlc_h5, xy_ci):
    x, _, _ = xy_ci
    coords = read_dlc(dlc_h5)["individual1"]
    np.testing.assert_allclose(coords["x"].to_numpy(), x)


def test_read_dlc_rejects_other_extensions(tmp_path):
    path = tmp_path / "recording.txt"
    path.write_text("not a pose file")
    with pytest.raises(ValueError):
        read_dlc(path)


def test_read_sleap_returns_arrays(sleap_h5, xy_ci):
    x, _, _ = xy_ci
    coords = read_sleap(sleap_h5)
    assert list(coords) == ["track0"]
    assert set(coords["track0"]) == COORD_KEYS
    np.testing.assert_allclose(coords["track0"]["x"], x)


def test_read_poser_coords_round_trips(tmp_path, xy_ci):
    x, y, ci = xy_ci
    written = save_coords_to_h5(
        {"1": {"x": x, "y": y, "ci": ci}}, tmp_path / "rec.avi"
    )
    coords = read_poser_coords(written)
    assert list(coords) == ["1"]
    np.testing.assert_allclose(coords["1"]["x"], x)
    np.testing.assert_allclose(coords["1"]["ci"], ci)


# --- format detection ------------------------------------------------------


def test_read_coords_detects_dlc(dlc_h5):
    assert list(read_coords(dlc_h5)) == ["individual1"]


def test_read_coords_detects_sleap(sleap_h5):
    assert list(read_coords(sleap_h5)) == ["track0"]


def test_read_coords_shortcuts_on_poser_coords_in_name(tmp_path, xy_ci):
    x, y, ci = xy_ci
    written = save_coords_to_h5(
        {"1": {"x": x, "y": y, "ci": ci}}, tmp_path / "rec.avi"
    )
    assert "poser_coords" in written
    assert list(read_coords(written)) == ["1"]


def test_read_coords_raises_when_no_reader_matches(tmp_path):
    path = tmp_path / "recording.txt"
    path.write_text("not a pose file")
    with pytest.raises(PoseFormatError) as excinfo:
        read_coords(path)
    assert isinstance(excinfo.value, PoseRError)


def test_read_coords_error_names_every_reader_tried(tmp_path):
    path = tmp_path / "recording.txt"
    path.write_text("not a pose file")
    with pytest.raises(PoseFormatError) as excinfo:
        read_coords(path)
    message = str(excinfo.value)
    assert "DeepLabCut" in message
    assert "SLEAP" in message
    assert "PoseR-native" in message


def test_read_coords_error_chains_the_last_failure(tmp_path):
    path = tmp_path / "recording.txt"
    path.write_text("not a pose file")
    with pytest.raises(PoseFormatError) as excinfo:
        read_coords(path)
    assert excinfo.value.__cause__ is not None


# --- classification round-trip --------------------------------------------


def test_classification_round_trip(tmp_path, classification_data, behaviour_schema):
    written = save_to_h5(
        classification_data, tmp_path / "rec.avi", V, behaviour_schema
    )
    back = read_classification_h5(written)
    assert list(back) == [1]
    assert back[1][1]["classification"] == "swim"
    assert back[1][2]["classification"] == "turn"


def test_classification_keys_are_one_based(
    tmp_path, classification_data, behaviour_schema
):
    # The file stores behaviour numbers 0 and 1; the reader returns 1 and 2.
    assert sorted(classification_data[1]) == [0, 1]
    written = save_to_h5(
        classification_data, tmp_path / "rec.avi", V, behaviour_schema
    )
    assert sorted(read_classification_h5(written)[1]) == [1, 2]


def test_classification_splits_coords_and_ci(
    tmp_path, classification_data, behaviour_schema
):
    written = save_to_h5(
        classification_data, tmp_path / "rec.avi", V, behaviour_schema
    )
    bout = read_classification_h5(written)[1][1]
    n = V * 10
    assert bout["coords"].shape == (n, 3)
    assert bout["ci"].shape == (n,)
    np.testing.assert_allclose(bout["coords"], classification_data[1][0]["coords"])


def test_save_to_h5_raises_on_malformed_bout(tmp_path, behaviour_schema):
    incomplete = {1: {0: {"classification": "swim", "coords": np.zeros((10, 3))}}}
    with pytest.raises(BehaviourWriteError) as excinfo:
        save_to_h5(incomplete, tmp_path / "rec.avi", V, behaviour_schema)
    assert excinfo.value.__cause__ is not None


def test_writers_name_output_after_the_video(tmp_path, xy_ci, classification_data, behaviour_schema):
    x, y, ci = xy_ci
    coords_path = save_coords_to_h5(
        {"1": {"x": x, "y": y, "ci": ci}}, tmp_path / "rec.avi"
    )
    cls_path = save_to_h5(
        classification_data, tmp_path / "rec.avi", V, behaviour_schema
    )
    assert coords_path.endswith("rec.avi_poser_coords.h5")
    assert cls_path.endswith("rec.avi_classification.h5")


# --- CTVM conversion -------------------------------------------------------


@pytest.mark.parametrize("fixture_name", ["dlc_h5", "dlc_csv"])
def test_convert_dlc_to_ctvm_shape(request, fixture_name):
    array = convert_dlc_to_ctvm(request.getfixturevalue(fixture_name))
    # (C, T, V, M) — M is always 1, multi-animal files are flattened.
    assert array.shape == (3, T, V, 1)


def test_convert_dlc_to_ctvm_rejects_other_extensions(tmp_path):
    path = tmp_path / "recording.txt"
    path.write_text("not a pose file")
    with pytest.raises(UnsupportedFormatError) as excinfo:
        convert_dlc_to_ctvm(path)
    assert isinstance(excinfo.value, PoseFormatError)
