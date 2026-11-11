"""Tests for poser.training.data_prep."""

from __future__ import annotations

import numpy as np
import pytest

from poser.training import data_prep
from poser.training.config import TrainingConfig
from poser.training.data_prep import prepare_dataset

NAMES = {0: "swim", 1: "turn"}


@pytest.fixture
def named_bouts(monkeypatch) -> None:
    """Six bouts named turn, swim, turn, ... with the HDF5 reading stubbed out."""
    names = np.array(["turn", "swim"] * 3)
    monkeypatch.setattr(data_prep, "read_classification_h5", lambda path: {})
    monkeypatch.setattr(
        data_prep,
        "classification_data_to_bouts",
        lambda data, **kwargs: (np.zeros((6, 3, 8, 4, 1)), names),
    )


def test_hdf5_behaviour_names_map_to_schema_indices(named_bouts):
    # The names went straight to PoseDataset, which casts labels to int64,
    # so a file labelled with behaviour names failed before training started.
    config = TrainingConfig(behaviour_schema={"labels": NAMES})

    train_loader, _ = prepare_dataset(["bouts.h5"], config)

    labels = train_loader.dataset.dataset.labels
    assert labels.dtype == np.int64
    assert labels.tolist() == [1, 0, 1, 0, 1, 0]


def test_hdf5_behaviour_name_missing_from_schema_is_rejected(named_bouts):
    config = TrainingConfig(behaviour_schema={"labels": {0: "swim"}})

    with pytest.raises(ValueError, match=r"\['turn'\].*behaviour_schema"):
        prepare_dataset(["bouts.h5"], config)
