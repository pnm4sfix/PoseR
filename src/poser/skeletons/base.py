"""
Base class for skeleton specifications.

Researchers who need custom graph logic beyond a plain YAML definition
can subclass :class:`BaseSkeletonSpec` and register it via the
``poser.skeletons`` setuptools entry-point group.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseSkeletonSpec(ABC):
    """Abstract skeleton specification.

    Subclass this to define a new species / context programmatically.
    For simple skeletons, prefer a YAML file instead.

    Required class attributes (or override as properties):

    .. code-block:: python

        class MySpecies(BaseSkeletonSpec):
            name = "myspecies"
            num_nodes = 10
            node_names = ["head", "neck", ..., "tail"]
            edges = [[0, 1], [1, 2], ...]
            center_node = 5
            head_node = 0
    """

    # Subclasses may set these as class-level attributes instead of overriding
    name: str = ""
    num_nodes: int = 0
    node_names: List[str] = []
    edges: List[List[int]] = []
    center_node: int = 0
    head_node: int = 0
    partition_strategy: str = "spatial"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check basic consistency of the skeleton spec.

        Raises
        ------
        ValueError
            If any constraint is violated.
        """
        if not self.name:
            raise ValueError("Skeleton spec must have a non-empty name.")
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if self.node_names and len(self.node_names) != self.num_nodes:
            raise ValueError(
                f"node_names has {len(self.node_names)} entries but "
                f"num_nodes = {self.num_nodes}."
            )
        for i, (a, b) in enumerate(self.edges):
            if a < 0 or a >= self.num_nodes or b < 0 or b >= self.num_nodes:
                raise ValueError(
                    f"Edge {i} ({a}, {b}) references invalid node index "
                    f"(num_nodes={self.num_nodes})."
                )
        if not (0 <= self.center_node < self.num_nodes):
            raise ValueError(
                f"center_node {self.center_node} is out of range [0, {self.num_nodes})."
            )
        if not (0 <= self.head_node < self.num_nodes):
            raise ValueError(
                f"head_node {self.head_node} is out of range [0, {self.num_nodes})."
            )

    def to_dict(self) -> dict:
        """Serialise to a YAML-compatible dict."""
        return {
            "name": self.name,
            "num_nodes": self.num_nodes,
            "node_names": self.node_names,
            "edges": [list(e) for e in self.edges],
            "center_node": self.center_node,
            "head_node": self.head_node,
            "partition_strategy": self.partition_strategy,
        }


class YAMLSkeletonSpec(BaseSkeletonSpec):
    """A skeleton spec loaded from a YAML file.  Not intended for subclassing."""

    def __init__(self, data: dict):
        self.name = data["name"]
        self.num_nodes = int(data["num_nodes"])
        self.node_names = data.get("node_names", [])
        self.edges = [list(e) for e in data.get("edges", [])]
        self.center_node = int(data.get("center_node", 0))
        self.head_node = int(data.get("head_node", 0))
        self.partition_strategy = data.get("partition_strategy", "spatial")

    def __repr__(self) -> str:
        return (
            f"YAMLSkeletonSpec(name={self.name!r}, "
            f"num_nodes={self.num_nodes}, "
            f"center={self.center_node})"
        )
