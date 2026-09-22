"""Typed data models shared across the core layer.

Modules here hold data and its validation, nothing else. They import stdlib,
pydantic and poser.core.exceptions, and never a sibling core module, so they
stay free of cycles as the rest of core grows models.

The training configuration lives here rather than under training/ so that core
modules can be typed against it without importing upward (STYLEGUIDE 1.1).
"""

from __future__ import annotations

from .batch import BatchMode, BatchResult
from .training import (
    AugmentationConfig,
    BehaviourSchema,
    DataConfig,
    ModelConfig,
    OptimiserConfig,
    TrainerConfig,
    TrainingConfig,
)

__all__ = [
    "AugmentationConfig",
    "BatchMode",
    "BatchResult",
    "BehaviourSchema",
    "DataConfig",
    "ModelConfig",
    "OptimiserConfig",
    "TrainerConfig",
    "TrainingConfig",
]
