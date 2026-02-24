"""
models/base.py
~~~~~~~~~~~~~~
Abstract base class that every PoseR model must extend.

Concrete models (e.g. ``ST_GCN_18``, ``C3D``) inherit from
:class:`BasePoseModel` and gain:

* a standard ``predict_step`` that returns labelled dicts
* ``save`` / ``load_from`` convenience wrappers
* model-card metadata helpers
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import lightning.pytorch as pl


class BasePoseModel(pl.LightningModule, abc.ABC):
    """Minimal contract every PoseR model must satisfy.

    Sub-classes must implement :meth:`forward` (and typically
    :meth:`training_step`, :meth:`configure_optimizers`).
    """

    # ------------------------------------------------------------------
    # Identity — fill in sub-class metadata
    # ------------------------------------------------------------------
    MODEL_NAME: str = "base"
    """Short unique identifier used in the model registry."""

    MODEL_VERSION: str = "0.0"
    """Semantic version string for the architecture."""

    SUPPORTED_INPUT_SHAPE: Optional[tuple] = None
    """Expected ``(C, T, V, M)`` input shape, or ``None`` if flexible."""

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape ``(N, C, T, V, M)``

        Returns
        -------
        logits : torch.Tensor, shape ``(N, num_classes)`` or ``(N, 1)``
        """

    # ------------------------------------------------------------------
    # Predict step (works without sub-class override)
    # ------------------------------------------------------------------
    def predict_step(
        self,
        batch: torch.Tensor | tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=-1) if logits.shape[-1] > 1 else torch.sigmoid(logits)
        preds = logits.argmax(dim=-1) if logits.shape[-1] > 1 else (probs > 0.5).long().squeeze(-1)
        return {"logits": logits, "probs": probs, "preds": preds}

    # ------------------------------------------------------------------
    # Convenience I/O
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Save model weights to *path* (`.pt` checkpoint)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    @classmethod
    def load_from(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        **model_kwargs: Any,
    ) -> "BasePoseModel":
        """Instantiate and load weights from *path*.

        Parameters
        ----------
        path:
            Path to a ``.pt`` state-dict or ``.ckpt`` Lightning checkpoint.
        model_kwargs:
            Forwarded to ``cls.__init__``.
        """
        path = Path(path)
        model = cls(**model_kwargs)
        if path.suffix == ".ckpt":
            ckpt = torch.load(path, map_location=map_location)
            model.load_state_dict(ckpt["state_dict"])
        else:
            state = torch.load(path, map_location=map_location)
            model.load_state_dict(state)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Model card metadata
    # ------------------------------------------------------------------
    def model_card(self) -> Dict[str, Any]:
        """Return a dict describing this model's architecture."""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "num_parameters": n_params,
            "num_trainable_parameters": n_trainable,
            "supported_input_shape": self.SUPPORTED_INPUT_SHAPE,
            "class": type(self).__qualname__,
        }

    def __repr__(self) -> str:  # pragma: no cover
        card = self.model_card()
        return (
            f"{card['class']}("
            f"name={card['model_name']!r}, "
            f"params={card['num_parameters']:,})"
        )
