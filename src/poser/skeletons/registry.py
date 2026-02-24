"""
Skeleton registry — discovers built-in YAMLs and external Python subclasses.

Usage::

    from poser.skeletons import get_skeleton, list_skeletons

    spec = get_skeleton("zebrafish")
    print(spec.num_nodes, spec.center_node)
    print(list_skeletons())
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .base import BaseSkeletonSpec, YAMLSkeletonSpec

_BUILTIN_DIR = Path(__file__).parent / "built_in"
_ENTRY_POINT_GROUP = "poser.skeletons"


class _SkeletonRegistry:
    """Singleton registry of all known skeleton specs."""

    def __init__(self):
        self._specs: Dict[str, BaseSkeletonSpec] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load_builtin_yamls()
        self._load_entry_points()

    def _load_builtin_yamls(self) -> None:
        for yaml_file in _BUILTIN_DIR.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                spec = YAMLSkeletonSpec(data)
                spec.validate()
                self._specs[spec.name] = spec
            except Exception as exc:
                print(f"[SkeletonRegistry] Could not load {yaml_file.name}: {exc}")

    def _load_entry_points(self) -> None:
        try:
            eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        except TypeError:
            # Python < 3.9
            all_eps = importlib.metadata.entry_points()
            eps = all_eps.get(_ENTRY_POINT_GROUP, [])

        for ep in eps:
            try:
                cls = ep.load()
                if isinstance(cls, type) and issubclass(cls, BaseSkeletonSpec):
                    instance = cls()
                    instance.validate()
                    self._specs[instance.name] = instance
                    print(f"[SkeletonRegistry] Registered entry-point skeleton: {instance.name}")
            except Exception as exc:
                print(f"[SkeletonRegistry] Could not load entry-point {ep.name!r}: {exc}")

    def register_yaml(self, yaml_path: str) -> BaseSkeletonSpec:
        """Load and register a YAML skeleton file at runtime.

        Parameters
        ----------
        yaml_path:
            Path to the YAML file.

        Returns
        -------
        BaseSkeletonSpec
            The registered spec.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        spec = YAMLSkeletonSpec(data)
        spec.validate()
        self._specs[spec.name] = spec
        return spec

    def register(self, spec: BaseSkeletonSpec) -> None:
        """Register a :class:`BaseSkeletonSpec` instance directly."""
        spec.validate()
        self._specs[spec.name] = spec

    def get(self, name: str) -> BaseSkeletonSpec:
        """Return the named skeleton spec.

        Parameters
        ----------
        name:
            Skeleton name, e.g. ``"zebrafish"``.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        self._ensure_loaded()
        if name not in self._specs:
            available = ", ".join(sorted(self._specs.keys()))
            raise KeyError(
                f"Unknown skeleton {name!r}. Available: {available}"
            )
        return self._specs[name]

    def list(self) -> List[str]:
        """Return sorted list of all registered skeleton names."""
        self._ensure_loaded()
        return sorted(self._specs.keys())

    def validate_yaml(self, yaml_path: str) -> None:
        """Validate a skeleton YAML file without registering it.

        Prints a summary on success, raises on failure.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        spec = YAMLSkeletonSpec(data)
        spec.validate()
        print(f"✓ Skeleton {spec.name!r} is valid.")
        print(f"  Nodes: {spec.num_nodes}")
        print(f"  Edges: {len(spec.edges)}")
        print(f"  Center node: {spec.center_node}")
        print(f"  Head node:   {spec.head_node}")
        print(f"  Strategy:    {spec.partition_strategy}")


# Singleton
SkeletonRegistry = _SkeletonRegistry()


# Convenience module-level functions
def get_skeleton(name: str) -> BaseSkeletonSpec:
    """Return the skeleton spec for *name*."""
    return SkeletonRegistry.get(name)


def list_skeletons() -> List[str]:
    """Return all registered skeleton names."""
    return SkeletonRegistry.list()
