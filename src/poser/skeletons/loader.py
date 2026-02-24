"""
skeletons/loader.py
~~~~~~~~~~~~~~~~~~~
Converts a :class:`BaseSkeletonSpec` into a :class:`~poser.models.graph.Graph`
adjacency matrix so it can be fed directly into ST-GCN / C3D model configs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from poser.skeletons.base import BaseSkeletonSpec


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def yaml_to_graph(spec: "BaseSkeletonSpec"):
    """Return a :class:`~poser.models.graph.Graph` built from *spec*.

    Parameters
    ----------
    spec:
        Any :class:`BaseSkeletonSpec` instance (YAML-loaded or Python-defined).

    Returns
    -------
    graph : poser.models.graph.Graph
        Graph object with adjacency matrix ``A`` of shape
        ``(3, num_nodes, num_nodes)``.
    """
    from poser.models.graph import Graph  # lazy — avoids circular imports

    graph = Graph(
        layout="list",
        strategy=spec.partition_strategy,
        edge_list=[[int(a), int(b)] for a, b in spec.edges],
        num_nodes_override=spec.num_nodes,
        center_node=spec.center_node,
    )
    return graph


def spec_to_adj(spec: "BaseSkeletonSpec", strategy: str | None = None) -> np.ndarray:
    """Return the normalised adjacency array ``A`` for *spec*.

    Parameters
    ----------
    spec:
        Any :class:`BaseSkeletonSpec` instance.
    strategy:
        Override ``spec.partition_strategy`` if provided.

    Returns
    -------
    A : np.ndarray, shape ``(K, V, V)``
        Adjacency matrix where *K* is the number of subsets for the chosen
        partition strategy and *V* == ``spec.num_nodes``.
    """
    from poser.models.graph import Graph

    graph = Graph(
        layout="list",
        strategy=strategy or spec.partition_strategy,
        edge_list=[[int(a), int(b)] for a, b in spec.edges],
        num_nodes_override=spec.num_nodes,
        center_node=spec.center_node,
    )
    return graph.A
