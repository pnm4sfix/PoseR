"""Runtime knobs, overridable from the environment.

Values here tune how PoseR runs rather than what it computes: anything that a
user might reasonably change per machine without changing their results.
Experiment parameters belong in TrainingConfig instead.

Every field is read from a POSER_-prefixed environment variable, so

    POSER_INFERENCE_BATCH_SIZE=64 poser batch ...

overrides the default for that run. Import the module-level ``settings``
rather than constructing Settings, so one object is shared process-wide.
"""

from __future__ import annotations

import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-overridable runtime settings."""

    model_config = SettingsConfigDict(
        env_prefix="POSER_",
        case_sensitive=False,
        extra="ignore",
    )

    inference_batch_size: int = Field(
        16,
        ge=1,
        description="Clips per forward pass during behaviour decoding. "
        "Lower it if inference runs out of GPU memory.",
    )

    device: str = Field(
        "auto",
        description="Torch device for inference. 'auto' takes cuda, then mps, "
        "then cpu. Name one explicitly to force it, which is worth doing if "
        "an op is missing from the mps backend.",
    )


settings = Settings()


def resolve_device() -> torch.device:
    """Return the device inference should run on.

    Honours settings.device, and on "auto" prefers cuda, then Apple's mps,
    then cpu. Picking cuda-or-cpu alone silently lands on cpu for every Apple
    Silicon machine, which costs roughly a factor of two.
    """
    requested = settings.device.strip().lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
