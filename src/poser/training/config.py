"""Re-exports the training configuration models from core.schemas.training.

The models moved down to the core layer so that core modules can be typed
against them without importing upward from core into training (STYLEGUIDE
section 1.1). This module keeps the original import path working.
"""

from __future__ import annotations

from poser.core.schemas.training import (
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
    "BehaviourSchema",
    "DataConfig",
    "ModelConfig",
    "OptimiserConfig",
    "TrainerConfig",
    "TrainingConfig",
]
