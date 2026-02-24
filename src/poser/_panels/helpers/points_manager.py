"""
_panels/helpers/points_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Feature 2 — efficient napari ``Points`` layer management.

The original ``_widget.py`` called ``layer.data = new_array`` on every
single frame, causing the entire layer to be re-rendered for each update.

:class:`PointsLayerManager` buffers changes and applies them in one
assignment per cycle, keeping the viewer responsive even for large videos.

Usage
-----
::

    mgr = PointsLayerManager(viewer)
    mgr.add_individual("ind1", color="red", n_nodes=9)
    mgr.update_frame("ind1", frame_idx=42, coords_xy=xy_array)
    # …accumulate many updates in a loop…
    mgr.flush()   # single layer.data = … assignment per individual
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PointsLayerManager:
    """Manages buffered updates to napari ``Points`` layers.

    Parameters
    ----------
    viewer:
        A ``napari.Viewer`` instance.
    """

    def __init__(self, viewer: Any):
        self._viewer = viewer
        # {individual_key: {"layer": Points, "buffer": dict[frame_idx, ndarray]}}
        self._entries: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Layer lifecycle
    # ------------------------------------------------------------------

    def add_individual(
        self,
        key: str,
        *,
        n_nodes: int,
        color: str = "red",
        size: float = 6.0,
        edge_width: float = 0.0,
        name: Optional[str] = None,
        symbol: str = "disc",
    ) -> Any:
        """Create (or replace) a Points layer for *key*.

        Parameters
        ----------
        key:
            Unique identifier string (e.g. ``"ind1"``).
        n_nodes:
            Number of skeleton nodes.
        color:
            Face colour for the points.
        size:
            Point diameter.
        name:
            Optional layer name override.

        Returns
        -------
        layer : napari.layers.Points
        """
        layer_name = name or f"pose_{key}"

        # Remove old layer if it exists
        self.remove_individual(key)

        # Placeholder empty array — shape ``(0, 2)``
        empty = np.zeros((0, 2), dtype=np.float32)
        layer = self._viewer.add_points(
            empty,
            name=layer_name,
            face_color=color,
            size=size,
            edge_width=edge_width,
            symbol=symbol,
        )
        with self._lock:
            self._entries[key] = {
                "layer": layer,
                "buffer": {},
                "n_nodes": n_nodes,
                "visible_frame": None,
            }
        return layer

    def remove_individual(self, key: str) -> None:
        """Remove a managed layer and its buffer."""
        with self._lock:
            entry = self._entries.pop(key, None)
        if entry is not None:
            try:
                self._viewer.layers.remove(entry["layer"])
            except Exception:
                pass

    def layer(self, key: str):
        """Return the napari ``Points`` layer for *key*."""
        return self._entries[key]["layer"]

    # ------------------------------------------------------------------
    # Buffered updates
    # ------------------------------------------------------------------

    def update_frame(
        self,
        key: str,
        frame_idx: int,
        coords_xy: np.ndarray,
    ) -> None:
        """Buffer a per-frame coordinate update (does NOT write to layer yet).

        Parameters
        ----------
        key:
            Individual key.
        frame_idx:
            Frame / time index.
        coords_xy:
            ``(V, 2)`` array of ``[x, y]`` coordinates in *image* space.
        """
        with self._lock:
            if key not in self._entries:
                raise KeyError(f"No layer registered for individual '{key}'.")
            self._entries[key]["buffer"][frame_idx] = np.asarray(coords_xy, dtype=np.float32)

    def buffer_all_frames(
        self,
        key: str,
        all_coords: np.ndarray,
    ) -> None:
        """Buffer coordinates for every frame at once.

        Parameters
        ----------
        key:
            Individual key.
        all_coords:
            ``(T, V, 2)`` array — one ``(V, 2)`` slice per frame.
        """
        all_coords = np.asarray(all_coords, dtype=np.float32)
        T = all_coords.shape[0]
        with self._lock:
            if key not in self._entries:
                raise KeyError(f"No layer registered for individual '{key}'.")
            for t in range(T):
                self._entries[key]["buffer"][t] = all_coords[t]

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self, key: Optional[str] = None) -> None:
        """Apply all buffered updates to the napari layer(s).

        This performs exactly ONE ``layer.data = …`` assignment per
        individual, regardless of how many frame updates were buffered.

        Parameters
        ----------
        key:
            If given, flush only this individual.  Otherwise flush all.
        """
        keys = [key] if key is not None else list(self._entries)
        with self._lock:
            for k in keys:
                entry = self._entries.get(k)
                if entry is None or not entry["buffer"]:
                    continue
                self._flush_one(entry)

    def _flush_one(self, entry: dict) -> None:
        """Internal: write buffered coords to ``entry['layer'].data``."""
        buf: Dict[int, np.ndarray] = entry["buffer"]
        if not buf:
            return

        n_nodes: int = entry["n_nodes"]
        frame_indices = sorted(buf)

        # Build napari 3-D points array: (N_total, 3) where col 0 = frame idx
        rows: List[np.ndarray] = []
        for fidx in frame_indices:
            xy = buf[fidx]  # (V, 2)
            frame_col = np.full((len(xy), 1), fidx, dtype=np.float32)
            rows.append(np.hstack([frame_col, xy[:, [1, 0]]]))  # napari: [z, y, x]

        all_points = np.vstack(rows)  # (T*V, 3)

        entry["layer"].data = all_points
        entry["buffer"].clear()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def show_frame(self, key: str, frame_idx: int) -> None:
        """Filter the layer so only *frame_idx* points are visible.

        Uses ``layer.shown`` mask for zero-copy display of a single frame.
        """
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return
        layer = entry["layer"]
        data = layer.data  # (N, 3)
        if data.shape[0] == 0:
            return
        shown = data[:, 0].astype(int) == frame_idx
        layer.shown = shown
        entry["visible_frame"] = frame_idx

    def show_all(self, key: str) -> None:
        """Show all points in the layer for *key*."""
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return
        layer = entry["layer"]
        layer.shown = np.ones(len(layer.data), dtype=bool)
        entry["visible_frame"] = None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def keys(self) -> List[str]:
        """Return registered individual keys."""
        return list(self._entries)

    def clear_buffer(self, key: Optional[str] = None) -> None:
        """Discard buffered (un-flushed) updates without applying them."""
        keys = [key] if key is not None else list(self._entries)
        with self._lock:
            for k in keys:
                if k in self._entries:
                    self._entries[k]["buffer"].clear()
