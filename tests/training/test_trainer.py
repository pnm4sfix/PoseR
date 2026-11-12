"""Tests for poser.training.trainer."""

from __future__ import annotations

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

from poser.models.st_gcn_aaai18_pylightning_3block import ST_GCN_18
from poser.training.config import TrainingConfig
from poser.training.trainer import PoseRTrainer


def test_auto_lr_reaches_the_optimiser(tmp_path, monkeypatch):
    # The finder writes its choice to model.learning_rate, but the built-in
    # model builds its optimiser from _optimiser_config, so fit trained at the
    # rate the config started with.
    config = TrainingConfig(
        optimiser={"auto_lr": True, "scheduler": "none"}, output_dir=tmp_path
    )
    model = ST_GCN_18.from_training_config(config)

    def choose_half(self, model, **kwargs):
        model.learning_rate = 0.5

    monkeypatch.setattr(Tuner, "lr_find", choose_half)
    monkeypatch.setattr(pl.Trainer, "fit", lambda self, *args, **kwargs: None)

    PoseRTrainer(config).fit(model, train_loader=None)

    assert model.configure_optimizers().defaults["lr"] == 0.5
