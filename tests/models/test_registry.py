"""Tests for poser.models.registry."""

from __future__ import annotations

import lightning.pytorch as pl
import pytest
import torch

import poser.models  # noqa: F401  registers the built-in architectures
from poser.models.base import BasePoseModel
from poser.models.registry import _registry, load_model
from poser.models.st_gcn_aaai18_pylightning_3block import ST_GCN_18

C = 3
V = 9
NUM_CLASS = 4


def _save_decoder(path, graph_cfg: dict) -> ST_GCN_18:
    """Save a decoder as the released ones were, without hyper_parameters."""
    torch.manual_seed(0)
    model = ST_GCN_18(in_channels=C, num_class=NUM_CLASS, graph_cfg=graph_cfg)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "pytorch-lightning_version": pl.__version__,
        },
        path,
    )
    return model


@pytest.fixture
def decoder_ckpt(tmp_path) -> str:
    """A 9-node decoder, a node count that has a named layout."""
    path = tmp_path / "decoder.ckpt"
    _save_decoder(path, {"layout": "zebrafish"})
    return str(path)


def test_load_model_loads_a_built_in_checkpoint(decoder_ckpt):
    # The call core.behaviour_decode makes: a checkpoint and nothing else.
    # ST_GCN_18 had no load_from, so this raised AttributeError.
    model = load_model("st_gcn_3block", decoder_ckpt, map_location="cpu")
    assert isinstance(model, ST_GCN_18)
    saved = torch.load(decoder_ckpt, weights_only=False)["state_dict"]
    torch.testing.assert_close(model.state_dict()["fcn.weight"], saved["fcn.weight"])
    bouts = torch.zeros(2, C, 20, V, 1)  # (N, C, T, V, M)
    assert model.eval()(bouts).shape == (2, NUM_CLASS)


def test_load_model_takes_its_shape_from_the_weights(decoder_ckpt):
    # The call PoseR.predict makes: ModelConfig fields, here the defaults,
    # which describe a different network from the one in the checkpoint.
    model = load_model(
        "st_gcn_3block",
        checkpoint=decoder_ckpt,
        num_class=2,
        num_nodes=9,
        in_channels=3,
        layout="auto",
    )
    assert model.fcn.out_channels == NUM_CLASS


def test_load_model_loads_a_skeleton_with_no_named_layout(tmp_path):
    # Seven nodes match no layout in poser.models.graph, and training with the
    # default layout "auto" produces exactly such decoders.
    path = tmp_path / "auto.ckpt"
    saved = _save_decoder(path, {"layout": "auto", "num_nodes_override": 7})

    model = load_model("st_gcn_3block", str(path))

    torch.testing.assert_close(model.state_dict()["A"], saved.state_dict()["A"])
    bouts = torch.zeros(2, C, 20, 7, 1)  # (N, C, T, V, M)
    assert model.eval()(bouts).shape == (2, NUM_CLASS)


def test_arch_info_describes_a_built_in_without_model_metadata():
    # ST_GCN_18 is a plain LightningModule and declares no MODEL_VERSION,
    # which made `poser model list` raise AttributeError.
    info = _registry.arch_info("st_gcn_3block")

    assert info["name"] == "st_gcn_3block"
    assert info["class"] == "ST_GCN_18"
    assert info["model_version"] == BasePoseModel.MODEL_VERSION
    assert info["supported_input_shape"] is None
