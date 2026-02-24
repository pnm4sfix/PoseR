"""
training/config.py
~~~~~~~~~~~~~~~~~~
Pydantic v2 model replacing the scattered ``decoder_config.yml`` parsing
in ``_widget.py::initialise_params``.

A :class:`TrainingConfig` can be created from:

* A YAML file: ``TrainingConfig.from_yaml("decoder_config.yml")``
* Keyword arguments: ``TrainingConfig(num_class=5, layout="zebrafish", ...)``
* Environment variables with the ``POSER_`` prefix.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Parameters governing how raw pose data is pre-processed."""

    fps: int = Field(25, ge=1, description="Frames per second of the source video.")
    T: int = Field(100, ge=4, description="Clip length in frames before padding.")
    T2: int = Field(100, ge=4, description="Padded clip length fed to the model.")
    C: int = Field(3, ge=2, le=3, description="Channels per node (2=xy, 3=xyc).")
    M: int = Field(1, ge=1, description="Max individuals per frame.")
    denominator: float = Field(100.0, gt=0.0, description="Normalisation divisor for coordinates.")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    center_data: bool = True
    align_data: bool = True
    T_method: Literal["pad", "interpolate"] = "pad"


class ModelConfig(BaseModel):
    """Architecture hyper-parameters."""

    architecture: str = Field("st_gcn_3block", description="Registered architecture name.")
    layout: str = Field("zebrafish", description="Skeleton layout name.")
    in_channels: int = Field(3, ge=1)
    num_class: int = Field(2, ge=2)
    num_nodes: int = Field(9, ge=2)
    edge_importance_weighting: bool = True
    dropout: float = Field(0.5, ge=0.0, le=1.0)

    @field_validator("num_class")
    @classmethod
    def at_least_two(cls, v: int) -> int:
        if v < 2:
            raise ValueError("num_class must be >= 2.")
        return v


class OptimiserConfig(BaseModel):
    """Optimiser and scheduler settings."""

    learning_rate: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(1e-4, ge=0.0)
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    warmup_epochs: int = Field(5, ge=0)
    auto_lr: bool = Field(False, description="Run LR-finder before training.")


class TrainerConfig(BaseModel):
    """PyTorch Lightning Trainer settings."""

    max_epochs: int = Field(200, ge=1)
    batch_size: int = Field(32, ge=1)
    val_split: float = Field(0.2, gt=0.0, lt=1.0)
    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"
    early_stopping_patience: int = Field(20, ge=1)
    gradient_clip_val: float = Field(1.0, ge=0.0)
    log_every_n_steps: int = Field(10, ge=1)
    seed: int = 42

    class_weights: Optional[List[float]] = Field(
        None,
        description="Per-class loss weights (length must equal num_class).",
    )


class AugmentationConfig(BaseModel):
    """Toggles and magnitudes for online data augmentation."""

    rotate: bool = True
    rotate_max_deg: float = 45.0
    jitter: bool = True
    jitter_sigma: float = 0.02
    scale: bool = True
    scale_range: tuple[float, float] = (0.8, 1.2)
    shear: bool = False
    shear_max: float = 0.1
    roll: bool = True
    fragment: bool = True


class BehaviourSchema(BaseModel):
    """Maps integer label indices to behaviour names."""

    labels: Dict[int, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sorted_keys(self) -> "BehaviourSchema":
        self.labels = dict(sorted(self.labels.items()))
        return self

    def num_classes(self) -> int:
        return len(self.labels)

    def label_list(self) -> List[str]:
        return [self.labels[k] for k in sorted(self.labels)]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    """Full PoseR training configuration.

    Examples
    --------
    Load from YAML::

        cfg = TrainingConfig.from_yaml("decoder_config.yml")

    Build programmatically::

        cfg = TrainingConfig(
            model=ModelConfig(num_class=5, layout="zebrafish"),
            behaviour_schema=BehaviourSchema(labels={0: "swim", 1: "freeze", ...}),
        )
    """

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimiser: OptimiserConfig = Field(default_factory=OptimiserConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    behaviour_schema: BehaviourSchema = Field(default_factory=BehaviourSchema)

    output_dir: Path = Field(Path("poser_runs"), description="Where to save checkpoints/logs.")
    run_name: str = Field("run", description="Sub-directory name for this run.")
    species: str = Field("zebrafish", description="Species tag (informational).")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load a ``TrainingConfig`` from a YAML file.

        Supports the legacy flat ``decoder_config.yml`` format as well as
        the new nested format produced by :meth:`to_yaml`.
        """
        import yaml

        path = Path(path)
        with open(path) as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # --- legacy flat format compatibility ---
        if "model" not in raw and "layout" in raw:
            raw = _migrate_legacy(raw)

        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> Path:
        """Serialise to a human-readable YAML file."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        with open(path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
        return path

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    def label_dict(self) -> Dict[int, str]:
        return self.behaviour_schema.labels


# ---------------------------------------------------------------------------
# Legacy migration helper
# ---------------------------------------------------------------------------

def _migrate_legacy(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the old flat ``decoder_config.yml`` to the nested structure."""
    migrated: Dict[str, Any] = {}

    # Model
    migrated["model"] = {
        "layout": raw.get("layout", "zebrafish"),
        "num_class": raw.get("num_class", 2),
        "num_nodes": raw.get("num_nodes", 9),
        "in_channels": raw.get("in_channels", 3),
        "dropout": raw.get("dropout", 0.5),
    }

    # Data
    migrated["data"] = {
        "fps": raw.get("fps", 25),
        "T": raw.get("T", 100),
        "T2": raw.get("T2", 100),
        "C": raw.get("C", 3),
        "M": raw.get("M", 1),
        "denominator": raw.get("denominator", 100.0),
        "confidence_threshold": raw.get("confidence_threshold", 0.5),
    }

    # Trainer
    migrated["trainer"] = {
        "max_epochs": raw.get("max_epochs", 200),
        "batch_size": raw.get("batch_size", 32),
    }

    # Optimiser
    migrated["optimiser"] = {
        "learning_rate": raw.get("learning_rate", raw.get("lr", 1e-3)),
        "weight_decay": raw.get("weight_decay", 1e-4),
    }

    # Behaviour schema
    label_dict = raw.get("label_dict", raw.get("behaviour_dict", {}))
    if isinstance(label_dict, dict):
        # May be {name: idx} or {idx: name} — normalise to {int: str}
        norm: Dict[int, str] = {}
        for k, v in label_dict.items():
            if isinstance(k, int):
                norm[k] = str(v)
            elif isinstance(v, int):
                norm[int(v)] = str(k)
        migrated["behaviour_schema"] = {"labels": norm}

    # Pass-through fields
    for key in ("output_dir", "run_name", "species"):
        if key in raw:
            migrated[key] = raw[key]

    return migrated
