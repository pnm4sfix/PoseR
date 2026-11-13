"""Tests for poser.core.dataset."""

from __future__ import annotations

import numpy as np
import pytest

from poser.core import augmentation
from poser.core.dataset import PoseDataset

SEED = 20260916
SWITCHES = ("rotate", "jitter", "scale", "shear", "roll", "fragment")


@pytest.fixture
def clips(rng) -> np.ndarray:
    """Synthetic pose clips, (N, C, T, V, M)."""
    return rng.normal(0, 1, (4, 3, 30, 5, 1)).astype(np.float32)


@pytest.mark.parametrize("enabled", ["rotate", "jitter", "scale", "shear"])
def test_augmentation_dict_applies_only_the_enabled_transform(clips, enabled):
    # Any switch being on used to apply rotate, jitter, scale and shear
    # together, so a switch that was off still changed the training data.
    flags = dict.fromkeys(SWITCHES, False)
    flags[enabled] = True
    dataset = PoseDataset(
        data=clips, labels=np.zeros(4), transform=["pad"], T=30, augmentation=flags
    )

    np.random.seed(SEED)
    sample, _ = dataset[0]
    np.random.seed(SEED)
    expected = getattr(augmentation, f"{enabled}_transform")(clips[0], 1)[0]

    np.testing.assert_allclose(sample.numpy(), expected.astype(np.float32))


def test_augment_flag_without_dict_applies_random_augmentation(clips):
    dataset = PoseDataset(
        data=clips, labels=np.zeros(4), transform=["pad"], T=30, augment=True
    )

    np.random.seed(SEED)
    sample, _ = dataset[0]
    np.random.seed(SEED)
    expected = augmentation.random_augmentation(clips[0])

    np.testing.assert_allclose(sample.numpy(), expected.astype(np.float32))
