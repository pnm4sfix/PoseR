"""
Session management — track multiple (pose_file, video_file) pairs and their
annotation state so an analyst can work through a queue without having to
reload the plugin between files.

Feature 4 of the redesign: session table / multi-layer annotation management.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SessionEntry:
    """Everything associated with one (pose_file, video_file) pair.

    Attributes
    ----------
    pose_path:
        Path to the raw pose estimation file (.h5 / .csv).
    video_path:
        Path to the paired video file.
    status:
        One of ``"loaded"``, ``"annotating"``, ``"done"``.
    coords_data:
        Decoded pose data dict (populated after loading).
    classification_data:
        Annotation dict (populated during annotation).
    layer_names:
        Set of napari layer names that belong to this entry.
    meta:
        Arbitrary extra metadata (fps, species, etc.).
    """

    pose_path: str
    video_path: str = ""
    status: str = "loaded"
    coords_data: Dict = field(default_factory=dict)
    classification_data: Dict = field(default_factory=dict)
    layer_names: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def stem(self) -> str:
        """Filename stem used for layer naming."""
        return Path(self.pose_path).stem

    def layer_name(self, kind: str) -> str:
        """Return the napari layer name for this entry and layer kind.

        Parameters
        ----------
        kind:
            One of ``"video"``, ``"points"``, ``"tracks"``.
        """
        return f"{self.stem}_{kind}"

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (omits large arrays)."""
        return {
            "pose_path": self.pose_path,
            "video_path": self.video_path,
            "status": self.status,
            "layer_names": self.layer_names,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionEntry":
        return cls(
            pose_path=d["pose_path"],
            video_path=d.get("video_path", ""),
            status=d.get("status", "loaded"),
            layer_names=d.get("layer_names", []),
            meta=d.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages a queue of :class:`SessionEntry` objects.

    Provides:
    * Add / remove entries.
    * Activate one entry at a time (hides other entries' napari layers).
    * Persist to / restore from ``session.json`` in the project directory.

    Parameters
    ----------
    project_dir:
        Directory where ``session.json`` will be written.
    viewer:
        Optional napari :class:`Viewer` reference used for layer visibility.
    """

    _SESSION_FILE = "session.json"

    def __init__(self, project_dir: str = "", viewer=None):
        self.project_dir = project_dir
        self.viewer = viewer
        self.entries: List[SessionEntry] = []
        self._active_idx: Optional[int] = None

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add(self, pose_path: str, video_path: str = "", meta: Optional[Dict] = None) -> SessionEntry:
        """Add a new entry to the session queue."""
        entry = SessionEntry(
            pose_path=pose_path,
            video_path=video_path,
            meta=meta or {},
        )
        self.entries.append(entry)
        self.save()
        return entry

    def remove(self, idx: int) -> None:
        """Remove entry *idx* from the queue."""
        if self._active_idx == idx:
            self._active_idx = None
        self.entries.pop(idx)
        self.save()

    def add_batch(self, pairs: List[tuple]) -> None:
        """Add multiple ``(pose_path, video_path)`` pairs at once."""
        for pair in pairs:
            pose = pair[0]
            vid = pair[1] if len(pair) > 1 else ""
            meta = pair[2] if len(pair) > 2 else {}
            self.add(pose, vid, meta)

    # ------------------------------------------------------------------
    # Activation (single-active model)
    # ------------------------------------------------------------------

    @property
    def active(self) -> Optional[SessionEntry]:
        """The currently active entry, or None."""
        if self._active_idx is None:
            return None
        return self.entries[self._active_idx]

    def activate(self, idx: int) -> SessionEntry:
        """Set entry *idx* as active.

        If a napari viewer is attached, hides all other entries' layers and
        makes this entry's layers visible.
        """
        if idx < 0 or idx >= len(self.entries):
            raise IndexError(f"No session entry at index {idx}")
        self._active_idx = idx
        entry = self.entries[idx]
        entry.status = "annotating"

        if self.viewer is not None:
            # Collect all layer names belonging to other entries
            other_names: set = set()
            for i, e in enumerate(self.entries):
                if i != idx:
                    other_names.update(e.layer_names)
            active_names = set(entry.layer_names)

            for layer in self.viewer.layers:
                if layer.name in active_names:
                    layer.visible = True
                elif layer.name in other_names:
                    layer.visible = False

        self.save()
        return entry

    def mark_done(self, idx: Optional[int] = None) -> None:
        """Mark *idx* (or the active entry) as done."""
        target = idx if idx is not None else self._active_idx
        if target is not None:
            self.entries[target].status = "done"
            self.save()

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def next_pending(self) -> Optional[int]:
        """Return the index of the next entry that is not ``"done"``."""
        for i, e in enumerate(self.entries):
            if e.status != "done":
                return i
        return None

    def all_done(self) -> bool:
        return all(e.status == "done" for e in self.entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write current session to ``session.json``."""
        if not self.project_dir:
            return
        path = os.path.join(self.project_dir, self._SESSION_FILE)
        data = {
            "entries": [e.to_dict() for e in self.entries],
            "active_idx": self._active_idx,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Restore session from ``session.json`` if it exists."""
        path = os.path.join(self.project_dir, self._SESSION_FILE)
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.entries = [SessionEntry.from_dict(d) for d in data.get("entries", [])]
        self._active_idx = data.get("active_idx")
        print(f"Session restored: {len(self.entries)} entries, "
              f"active={self._active_idx}")

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [f"SessionManager({len(self.entries)} entries):"]
        for i, e in enumerate(self.entries):
            marker = "→" if i == self._active_idx else " "
            lines.append(f"  {marker} [{i}] {e.status:12s} {e.stem}")
        return "\n".join(lines)
