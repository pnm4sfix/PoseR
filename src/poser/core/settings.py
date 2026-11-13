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

import logging
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

# The data_dir subfolder each kind of generated file goes to by default.
OutputKind = Literal[
    "batch",
    "finetune",
    "frames",
    "logs",
    "metrics",
    "models",
    "pose",
    "predictions",
    "runs",
    "sessions",
    "training",
]


def _find_data_dir(package_dir: Path) -> Path:
    """Return <repository>/data for a source checkout, else ~/.poser/data."""
    repository = package_dir.parents[1]  # package_dir is <repository>/src/poser
    if (repository / "pyproject.toml").is_file():
        return repository / "data"
    return Path.home() / ".poser" / "data"


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

    data_dir: Path = Field(
        default_factory=lambda: _find_data_dir(Path(__file__).resolve().parents[1]),
        description="Folder generated files default to, one subfolder per kind. "
        "In a source checkout it is the repository's data/. An installed "
        "package has no repository, so it is ~/.poser/data.",
    )

    log_level: str = Field(
        "INFO",
        description="Lowest level of PoseR log messages shown in the terminal: "
        "DEBUG, INFO, WARNING or ERROR. The log file in data_dir/logs keeps "
        "everything from DEBUG up.",
    )

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        """Accept any case, so POSER_LOG_LEVEL=debug works."""
        level = value.upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(
                f"log_level must be DEBUG, INFO, WARNING or ERROR, not {value!r}"
            )
        return level


settings = Settings()


def resolve_device() -> torch.device:
    """Return the device inference should run on.

    Honours settings.device, and on "auto" prefers cuda, then Apple's mps,
    then cpu. Picking cuda-or-cpu alone silently lands on cpu for every Apple
    Silicon machine, which costs roughly a factor of two.
    """
    requested = settings.device.strip().lower()
    if requested != "auto":
        log.info("Using device %s, as POSER_DEVICE asks", requested)
        return torch.device(requested)
    if torch.cuda.is_available():
        log.info("Using device cuda: it is available")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        log.info("Using device mps: CUDA is not available, MPS is")
        return torch.device("mps")
    log.info("Using device cpu: neither CUDA nor MPS is available")
    return torch.device("cpu")


def resolve_output_dir(kind: OutputKind) -> Path:
    """Return the folder this kind of output defaults to, creating it if missing."""
    folder = settings.data_dir / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder
