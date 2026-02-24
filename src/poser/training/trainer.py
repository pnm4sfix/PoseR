"""
training/trainer.py
~~~~~~~~~~~~~~~~~~~
``PoseRTrainer`` — a thin wrapper around ``lightning.Trainer`` that wires up
standard callbacks (EarlyStopping, ModelCheckpoint, LR monitoring) and
integrates :class:`TrainingConfig`.

Usage
-----
::

    from poser.training import TrainingConfig, PoseRTrainer

    cfg = TrainingConfig.from_yaml("decoder_config.yml")
    trainer = PoseRTrainer(cfg)
    trainer.fit(model, train_loader, val_loader)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from poser.training.config import TrainingConfig

log = logging.getLogger(__name__)


class PoseRTrainer:
    """Convenience wrapper around :class:`lightning.pytorch.Trainer`.

    Parameters
    ----------
    config:
        A :class:`TrainingConfig` instance.  Use
        ``TrainingConfig.from_yaml(path)`` to load from YAML.
    extra_callbacks:
        Additional Lightning callbacks to add on top of the defaults.
    logger:
        A Lightning logger instance.  Defaults to ``TensorBoardLogger`` in
        ``config.run_dir``.
    """

    def __init__(
        self,
        config: TrainingConfig,
        extra_callbacks: Optional[list] = None,
        logger=None,
    ):
        self.config = config
        self._extra_callbacks = extra_callbacks or []
        self._logger = logger
        self._trainer: Optional[pl.Trainer] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        model: pl.LightningModule,
        train_loader,
        val_loader=None,
        *,
        ckpt_path: Optional[str | Path] = None,
    ) -> pl.Trainer:
        """Train *model* using *train_loader* (and optional *val_loader*).

        Parameters
        ----------
        model:
            Any :class:`~lightning.pytorch.LightningModule`.
        train_loader:
            A :class:`torch.utils.data.DataLoader` for training data.
        val_loader:
            Optional validation dataloader.
        ckpt_path:
            Resume from a checkpoint.

        Returns
        -------
        trainer : pl.Trainer
            The configured trainer after training completes.
        """
        trainer = self._build_trainer()
        self._trainer = trainer

        if self.config.optimiser.auto_lr:
            from lightning.pytorch.tuner import Tuner
            tuner = Tuner(trainer)
            tuner.lr_find(model, train_dataloaders=train_loader)
            log.info("Auto LR: new learning_rate = %.2e", model.hparams.learning_rate)

        pl.seed_everything(self.config.trainer.seed, workers=True)
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=str(ckpt_path) if ckpt_path else None,
        )
        return trainer

    def test(self, model: pl.LightningModule, test_loader) -> list:
        """Run test loop — requires a prior :meth:`fit` call."""
        if self._trainer is None:
            self._trainer = self._build_trainer()
        return self._trainer.test(model, dataloaders=test_loader)

    def predict(self, model: pl.LightningModule, dataloader) -> list:
        """Run predict loop."""
        if self._trainer is None:
            self._trainer = self._build_trainer()
        return self._trainer.predict(model, dataloaders=dataloader)

    @property
    def best_checkpoint(self) -> Optional[Path]:
        """Path to the best checkpoint file saved by ModelCheckpoint."""
        if self._trainer is None:
            return None
        for cb in self._trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                return Path(cb.best_model_path)
        return None

    # ------------------------------------------------------------------
    # Internal builder
    # ------------------------------------------------------------------

    def _build_trainer(self) -> pl.Trainer:
        tcfg = self.config.trainer
        run_dir = self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                dirpath=str(run_dir / "checkpoints"),
                filename="best-{epoch:03d}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=tcfg.early_stopping_patience,
                mode="min",
            ),
            LearningRateMonitor(logging_interval="epoch"),
            *self._extra_callbacks,
        ]

        # Default logger: TensorBoard
        if self._logger is None:
            try:
                from lightning.pytorch.loggers import TensorBoardLogger
                logger = TensorBoardLogger(
                    save_dir=str(self.config.output_dir),
                    name=self.config.run_name,
                )
            except Exception:
                logger = True  # use default CSV logger
        else:
            logger = self._logger

        return pl.Trainer(
            max_epochs=tcfg.max_epochs,
            accelerator=tcfg.accelerator,
            devices=tcfg.devices,
            precision=tcfg.precision,
            gradient_clip_val=tcfg.gradient_clip_val,
            log_every_n_steps=tcfg.log_every_n_steps,
            callbacks=callbacks,
            logger=logger,
            default_root_dir=str(run_dir),
        )
