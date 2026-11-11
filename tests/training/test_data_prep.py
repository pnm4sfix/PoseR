"""Tests for poser.training.data_prep."""

from __future__ import annotations

import numpy as np
import pytest
import tables as tb
import torch

from poser.core.io import save_to_h5
from poser.training import data_prep
from poser.training.config import TrainingConfig
from poser.training.data_prep import prepare_dataset

NAMES = {0: "swim", 1: "turn"}
V = 5
T = 10
CENTER_NODE = 3
HEAD_NODE = 2
WINDOW = 20


class Behaviour(tb.IsDescription):
    """The label table of a classification .h5, mirroring _widget.Behaviour."""

    number = tb.Int32Col()
    classification = tb.StringCol(16)
    n_nodes = tb.Int32Col()
    start = tb.Int32Col()
    stop = tb.Int32Col()


@pytest.fixture
def classification_h5(tmp_path) -> str:
    """A classification .h5 with one swim and one turn bout, T frames of V nodes."""
    rng = np.random.default_rng(20260916)
    bouts = {
        behaviour: {
            "classification": name,
            "coords": rng.normal(5, 1, (V * T, 3)),
            "ci": np.full(V * T, 0.9),
            "start": behaviour * 25,
            "stop": behaviour * 25 + T,
        }
        for behaviour, name in NAMES.items()
    }
    return save_to_h5({1: bouts}, tmp_path / "rec.avi", V, Behaviour)


def _hdf5_bouts(path: str, **data) -> np.ndarray:
    """The (N, C, T, V, M) bouts prepare_dataset builds from an HDF5 file."""
    config = TrainingConfig(
        data={"T": T, "T2": T, **data},
        model={"num_nodes": V},
        behaviour_schema={"labels": NAMES},
    )
    train_loader, _ = prepare_dataset([path], config)
    return train_loader.dataset.dataset.data


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


def test_hdf5_bouts_are_centred_on_the_configured_node(classification_h5):
    # The on/off flag center_data was passed where a node index is expected,
    # which raised ValueError for every HDF5 file.
    bouts = _hdf5_bouts(
        classification_h5, center_data=True, center_node=CENTER_NODE, align_data=False
    )
    assert np.abs(bouts[:, 0:2, :, CENTER_NODE]).max() == 0
    assert np.abs(bouts[:, 0:2, :, 0]).max() > 0


def test_hdf5_bouts_are_not_centred_when_centring_is_off(classification_h5):
    bouts = _hdf5_bouts(
        classification_h5, center_data=False, center_node=CENTER_NODE, align_data=False
    )
    assert np.abs(bouts[:, 0:2, :, CENTER_NODE]).min() > 0


def test_hdf5_bouts_are_aligned_on_the_configured_head_node(classification_h5):
    bouts = _hdf5_bouts(
        classification_h5,
        center_data=True,
        center_node=CENTER_NODE,
        align_data=True,
        head_node=HEAD_NODE,
    )
    # Alignment turns the head node to point straight up at frame 0.
    assert bouts[:, 0, 0, HEAD_NODE, 0] == pytest.approx(0, abs=1e-5)
    assert (bouts[:, 1, 0, HEAD_NODE, 0] > 0).all()


def test_hdf5_bouts_use_the_node_count_stored_in_the_file(classification_h5):
    # config.model.num_nodes is only a default. The flat coords were reshaped
    # with it, which fails or mixes nodes and frames for any other skeleton.
    config = TrainingConfig(
        data={"T": T, "T2": T}, behaviour_schema={"labels": NAMES}
    )
    assert config.model.num_nodes != V

    train_loader, _ = prepare_dataset([classification_h5], config)

    assert train_loader.dataset.dataset.data.shape == (2, 3, T, V, 1)
    assert config.model.num_nodes == V


@pytest.fixture
def pose_npy(tmp_path):
    """One continuous recording, (C, T_total, V, M), with its per-frame labels."""
    rng = np.random.default_rng(20261111)
    pose = rng.normal(0, 1, (3, 200, V, 1))
    pose[2] = 0.9
    np.save(tmp_path / "rec_pose.npy", pose)
    np.save(tmp_path / "rec_labels.npy", rng.integers(0, 2, 200))
    return tmp_path / "rec_pose.npy"


def test_validation_inputs_are_the_same_on_every_pass(pose_npy):
    # Validation shared the training datasets, so every read was randomly
    # rotated, jittered and scaled and an unchanged model scored differently.
    config = TrainingConfig(data={"T": WINDOW, "T2": WINDOW})
    _, val_loader = prepare_dataset([pose_npy], config)

    first = torch.cat([x for x, _ in val_loader])
    second = torch.cat([x for x, _ in val_loader])

    assert torch.equal(first, second)


def test_training_inputs_are_still_augmented(pose_npy):
    config = TrainingConfig(data={"T": WINDOW, "T2": WINDOW})
    train_loader, _ = prepare_dataset([pose_npy], config)
    train_ds = train_loader.dataset

    assert not torch.equal(train_ds[0][0], train_ds[0][0])


def test_hdf5_training_bouts_are_augmented(classification_h5):
    # PoseDataset augments inside its transform branch, and the HDF5 dataset
    # was built without a transform list, so the switches did nothing for it.
    config = TrainingConfig(
        data={"T": T, "T2": T},
        model={"num_nodes": V},
        behaviour_schema={"labels": NAMES},
    )
    train_loader, val_loader = prepare_dataset([classification_h5], config)
    train_ds, val_ds = train_loader.dataset, val_loader.dataset

    assert not torch.equal(train_ds[0][0], train_ds[0][0])
    assert torch.equal(val_ds[0][0], val_ds[0][0])
