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
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from poser.training.config import TrainingConfig

log = logging.getLogger(__name__)


class _PrintEpochCallback(Callback):
    """Prints a plain-text epoch summary to stdout after every train epoch.

    Uses ``print(..., flush=True)`` so output streams immediately through
    pipes (e.g. the napari training panel subprocess).
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: N802
        metrics = trainer.callback_metrics
        parts = [f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}"]
        for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        # Fallback: print whatever is available
        if len(parts) == 1:
            for k, v in metrics.items():
                try:
                    parts.append(f"{k}={float(v):.4f}")
                except (TypeError, ValueError):
                    pass
        print("  ".join(parts), flush=True)


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
            Optional validation dataloader.  When ``None``, callbacks that
            monitor ``val_loss`` are automatically switched to ``train_loss``
            so the trainer does not crash or silently skip checkpointing.
        ckpt_path:
            Resume from a checkpoint.

        Returns
        -------
        trainer : pl.Trainer
            The configured trainer after training completes.
        """
        trainer = self._build_trainer(has_val=val_loader is not None)
        self._trainer = trainer

        if self.config.optimiser.auto_lr:
            from lightning.pytorch.tuner import Tuner
            # Store external loaders on model so val_dataloader() / train_dataloader()
            # hooks return them if Lightning's LR-finder bypasses external sources.
            if getattr(model, "_external_dataloaders", False):
                model._ext_train_dl = train_loader
                model._ext_val_dl = val_loader
            tuner = Tuner(trainer)
            tuner.lr_find(model, train_dataloaders=train_loader,
                          val_dataloaders=val_loader)
            log.info("Auto LR: new learning_rate = %.2e", model.learning_rate)

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
            self._trainer = self._build_trainer(has_val=False)
        return self._trainer.test(model, dataloaders=test_loader)

    def predict(self, model: pl.LightningModule, dataloader) -> list:
        """Run predict loop."""
        if self._trainer is None:
            self._trainer = self._build_trainer(has_val=False)
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

    def _build_trainer(self, *, has_val: bool = True) -> pl.Trainer:
        tcfg = self.config.trainer
        run_dir = self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        import torch
        torch.set_float32_matmul_precision("medium")  # use Tensor Cores if available

        # When there is no validation set, fall back to monitoring train_loss
        # so ModelCheckpoint and EarlyStopping don't crash or silently skip.
        monitor_metric = "val_loss" if has_val else "train_loss"
        ckpt_filename = (
            "best-{epoch:03d}-{val_loss:.4f}"
            if has_val
            else "best-{epoch:03d}-{train_loss:.4f}"
        )
        log.info(
            "Checkpoint / EarlyStopping monitor: %s%s",
            monitor_metric,
            "" if has_val else " (no val set — using train_loss)",
        )

        callbacks = [
            _PrintEpochCallback(),
            ModelCheckpoint(
                dirpath=str(run_dir / "checkpoints"),
                filename=ckpt_filename,
                monitor=monitor_metric,
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor=monitor_metric,
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
            enable_progress_bar=False,   # tqdm overwrites via \r — invisible in a Qt pipe
            enable_model_summary=True,
        )
