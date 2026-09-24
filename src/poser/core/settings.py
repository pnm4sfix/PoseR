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


settings = Settings()
