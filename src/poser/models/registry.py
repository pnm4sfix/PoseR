"""
models/registry.py
~~~~~~~~~~~~~~~~~~
Lightweight model registry for PoseR.

By default the registry discovers:

1. Built-in architectures registered via :func:`register_model`.
2. Any ``.ckpt`` / ``.pt`` weight files in the current working directory and
   ``~/.poser/models/``.
3. Third-party models declared via the ``poser.models`` entry-point group.

Usage
-----
::

    from poser.models.registry import list_models, load_model

    print(list_models())
    model = load_model("st_gcn_3block", checkpoint="path/to/weights.ckpt",
                       num_class=5, num_nodes=9)
"""
from __future__ import annotations

import importlib
import importlib.metadata
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from poser.models.base import BasePoseModel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry type aliases
# ---------------------------------------------------------------------------
_ArchClass = Type[BasePoseModel]
_ModelEntry = Dict[str, Any]  # {"class": cls, "description": str}


class _ModelRegistry:
    """Singleton model registry."""

    _instance: Optional["_ModelRegistry"] = None

    def __new__(cls) -> "_ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._archs: Dict[str, _ModelEntry] = {}
            cls._instance._checkpoints: List[Path] = []
            cls._instance._loaded_entry_points = False
        return cls._instance

    # ------------------------------------------------------------------
    # Architecture registration
    # ------------------------------------------------------------------
    def register(
        self,
        arch_class: _ArchClass,
        *,
        name: Optional[str] = None,
        description: str = "",
    ) -> _ArchClass:
        """Register an architecture class.

        Parameters
        ----------
        arch_class:
            A sub-class of :class:`BasePoseModel`.
        name:
            Override ``arch_class.MODEL_NAME`` as the registry key.
        description:
            Human-readable description of the architecture.
        """
        key = (name or arch_class.MODEL_NAME).lower()
        self._archs[key] = {"class": arch_class, "description": description}
        return arch_class

    def get_class(self, name: str) -> _ArchClass:
        """Return the architecture class registered under *name*."""
        self._ensure_entry_points()
        key = name.lower()
        if key not in self._archs:
            available = list(self._archs)
            raise KeyError(
                f"Unknown model architecture '{name}'. "
                f"Available: {available}"
            )
        return self._archs[key]["class"]

    def list_archs(self) -> List[str]:
        """Return sorted list of registered architecture names."""
        self._ensure_entry_points()
        return sorted(self._archs)

    def arch_info(self, name: str) -> Dict[str, Any]:
        """Return registration metadata for *name*."""
        self._ensure_entry_points()
        key = name.lower()
        if key not in self._archs:
            raise KeyError(f"Architecture '{name}' not registered.")
        entry = self._archs[key]
        cls: _ArchClass = entry["class"]
        return {
            "name": key,
            "class": cls.__qualname__,
            "description": entry["description"],
            "model_version": cls.MODEL_VERSION,
            "supported_input_shape": cls.SUPPORTED_INPUT_SHAPE,
        }

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------
    def scan_directory(self, directory: str | Path) -> List[Path]:
        """Scan *directory* for ``.ckpt`` and ``.pt`` weight files."""
        directory = Path(directory)
        if not directory.is_dir():
            return []
        found = list(directory.glob("**/*.ckpt")) + list(directory.glob("**/*.pt"))
        self._checkpoints = sorted(set(self._checkpoints + found))
        return found

    def list_checkpoints(self) -> List[Path]:
        """Return all discovered checkpoint paths."""
        # Always scan project dir + user home
        self.scan_directory(Path.cwd())
        self.scan_directory(Path.home() / ".poser" / "models")
        return list(self._checkpoints)

    # ------------------------------------------------------------------
    # Model instantiation
    # ------------------------------------------------------------------
    def load(
        self,
        arch: str,
        checkpoint: Optional[str | Path] = None,
        *,
        map_location: str = "cpu",
        **model_kwargs: Any,
    ) -> BasePoseModel:
        """Instantiate architecture *arch* and optionally load *checkpoint*.

        Parameters
        ----------
        arch:
            Registered architecture name (e.g. ``"st_gcn_3block"``).
        checkpoint:
            Path to a ``.ckpt`` or ``.pt`` weight file.
        model_kwargs:
            Constructor keyword arguments forwarded to the model class.
        """
        cls = self.get_class(arch)
        if checkpoint is not None:
            model = cls.load_from(checkpoint, map_location=map_location, **model_kwargs)
        else:
            model = cls(**model_kwargs)
        return model

    # ------------------------------------------------------------------
    # Entry-point discovery
    # ------------------------------------------------------------------
    def _ensure_entry_points(self) -> None:
        if self._loaded_entry_points:
            return
        self._loaded_entry_points = True
        try:
            eps = importlib.metadata.entry_points(group="poser.models")
        except Exception:
            return
        for ep in eps:
            try:
                cls = ep.load()
                if isinstance(cls, type) and issubclass(cls, BasePoseModel):
                    self.register(cls, description=f"entry-point: {ep.value}")
                    log.debug("Loaded model entry-point: %s → %s", ep.name, ep.value)
            except Exception as exc:
                log.warning("Failed to load model entry-point '%s': %s", ep.name, exc)


# ---------------------------------------------------------------------------
# Module-level singleton + convenience decorators
# ---------------------------------------------------------------------------
_registry = _ModelRegistry()


def register_model(
    name: Optional[str] = None,
    *,
    description: str = "",
) -> Callable[[_ArchClass], _ArchClass]:
    """Class decorator that registers an architecture with the registry.

    Example
    -------
    ::

        @register_model("my_arch", description="Custom 2-block ST-GCN")
        class MyArch(BasePoseModel):
            ...
    """
    def _decorator(cls: _ArchClass) -> _ArchClass:
        return _registry.register(cls, name=name, description=description)
    return _decorator


def list_models() -> List[str]:
    """Return sorted list of all registered architecture names."""
    return _registry.list_archs()


def load_model(
    arch: str,
    checkpoint: Optional[str | Path] = None,
    **kwargs: Any,
) -> BasePoseModel:
    """Instantiate *arch* and (optionally) load *checkpoint*.

    See :meth:`_ModelRegistry.load` for full parameter docs.
    """
    return _registry.load(arch, checkpoint, **kwargs)


def list_checkpoints() -> List[Path]:
    """Return all ``.ckpt``/``.pt`` files found in the project + user dirs."""
    return _registry.list_checkpoints()
