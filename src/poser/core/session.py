"""
Session management — track multiple (pose_file, video_file) pairs and their
annotation state so an analyst can work through a queue without having to
reload the plugin between files.

Feature 4 of the redesign: session table / multi-layer annotation management.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Viewer-keyed singleton registry
# ---------------------------------------------------------------------------

_SESSION_REGISTRY: Dict[int, "SessionManager"] = {}


def get_session(viewer) -> "SessionManager":
    """Return the shared :class:`SessionManager` for *viewer*.

    Creates one on first call and caches it by ``id(viewer)`` so every panel
    always operates on the same session.
    """
    key = id(viewer)
    log.debug(
        "Session for viewer %s: %s",
        key,
        "reused" if key in _SESSION_REGISTRY else "created",
    )
    if key not in _SESSION_REGISTRY:
        _SESSION_REGISTRY[key] = SessionManager(viewer=viewer)
    return _SESSION_REGISTRY[key]


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

    pose_path: str = ""
    video_path: str = ""
    status: str = "pending"
    coords_data: Dict = field(default_factory=dict)
    classification_data: Dict = field(default_factory=dict)
    layer_names: List[str] = field(default_factory=list)
    bouts: List = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def stem(self) -> str:
        """Filename stem used for layer naming."""
        if self.pose_path:
            return Path(self.pose_path).stem
        if self.video_path:
            return Path(self.video_path).stem
        return "entry"

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
            pose_path=str(pose_path) if pose_path else "",
            video_path=str(video_path) if video_path else "",
            meta=meta or {},
        )
        self.entries.append(entry)
        log.info(
            "Added entry %d (%s): pose %s, video %s",
            len(self.entries) - 1,
            entry.stem,
            entry.pose_path or "(none)",
            entry.video_path or "(none)",
        )
        self.save()
        return entry

    def remove(self, idx: int) -> None:
        """Remove entry *idx* from the queue."""
        log.debug(
            "Removing entry %s of %d (active index %s)",
            idx,
            len(self.entries),
            self._active_idx,
        )
        if self._active_idx == idx:
            self._active_idx = None
        elif self._active_idx is not None and 0 <= idx < self._active_idx:
            # Every entry after idx moves down one place.
            self._active_idx -= 1
        removed = self.entries.pop(idx)
        log.info(
            "Removed entry %s (%s); %d left, active index now %s",
            idx,
            removed.stem,
            len(self.entries),
            self._active_idx,
        )
        self.save()

    def add_batch(self, pairs: List[tuple]) -> None:
        """Add multiple ``(pose_path, video_path)`` pairs at once."""
        n_before = len(self.entries)
        for pair in pairs:
            pose = pair[0]
            vid = pair[1] if len(pair) > 1 else ""
            meta = pair[2] if len(pair) > 2 else {}
            self.add(pose, vid, meta)
        log.debug(
            "Batch added %d entries; the session has %d",
            len(self.entries) - n_before,
            len(self.entries),
        )

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
        log.debug(
            "Activating entry %s of %d (active index was %s)",
            idx,
            len(self.entries),
            self._active_idx,
        )
        if idx < 0 or idx >= len(self.entries):
            raise IndexError(f"No session entry at index {idx}")
        self._active_idx = idx
        entry = self.entries[idx]
        entry.status = "active"

        if self.viewer is not None:
            # Collect all layer names belonging to other entries
            other_names: set = set()
            for i, e in enumerate(self.entries):
                if i != idx:
                    other_names.update(e.layer_names)
            active_names = set(entry.layer_names)
            log.debug(
                "Showing layers %s; hiding the %d layer(s) of other entries",
                entry.layer_names,
                len(other_names),
            )

            for layer in self.viewer.layers:
                if layer.name in active_names:
                    layer.visible = True
                elif layer.name in other_names:
                    layer.visible = False

        log.info(
            "Activated entry %s (%s): pose %s, video %s, %d bouts, "
            "coords for %d individual(s)",
            idx,
            entry.stem,
            entry.pose_path or "(none)",
            entry.video_path or "(none)",
            len(entry.bouts),
            len(entry.coords_data),
        )
        self.save()
        return entry

    def mark_done(self, idx: Optional[int] = None) -> None:
        """Mark *idx* (or the active entry) as done."""
        log.debug("Mark done: index %s, active index %s", idx, self._active_idx)
        target = idx if idx is not None else self._active_idx
        if target is not None:
            self.entries[target].status = "done"
            log.info("Marked entry %s (%s) done", target, self.entries[target].stem)
            self.save()
        else:
            log.warning("Mark done: no index given and no active entry; nothing marked")

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def next_pending(self) -> Optional[int]:
        """Return the index of the next entry that is not ``"done"``."""
        for i, e in enumerate(self.entries):
            if e.status != "done":
                log.debug("Next pending entry: %d (%s, %s)", i, e.stem, e.status)
                return i
        log.debug("No pending entry: all %d entries are done", len(self.entries))
        return None

    def all_done(self) -> bool:
        log.debug(
            "All done check: %d of %d entries done",
            sum(e.status == "done" for e in self.entries),
            len(self.entries),
        )
        return all(e.status == "done" for e in self.entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path=None) -> None:
        """Write current session to *path* or ``session.json`` in project_dir."""
        if path is None:
            if not self.project_dir:
                log.debug("Session not autosaved: no project_dir set")
                return
            path = os.path.join(self.project_dir, self._SESSION_FILE)
        data = {
            "entries": [e.to_dict() for e in self.entries],
            "active_idx": self._active_idx,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(
            "Saved session to %s: %d entries, active=%s",
            path,
            len(self.entries),
            self._active_idx,
        )
        log.debug(
            "Session files keep no bouts or coords; bouts of %d entries not saved",
            sum(len(e.bouts) > 0 for e in self.entries),
        )

    def load(self, path=None) -> None:
        """Restore session from *path* or ``session.json`` in project_dir."""
        if path is None:
            path = os.path.join(self.project_dir, self._SESSION_FILE)
        log.debug(
            "Loading session from %s (replacing %d entries)", path, len(self.entries)
        )
        if not os.path.exists(str(path)):
            log.warning("Session file %s does not exist; nothing loaded", path)
            return
        with open(path) as f:
            data = json.load(f)
        self.entries = [SessionEntry.from_dict(d) for d in data.get("entries", [])]
        self._active_idx = data.get("active_idx")
        log.info(
            "Session restored from %s: %d entries, active=%s",
            path,
            len(self.entries),
            self._active_idx,
        )
        log.debug("Restored entries have no bouts or coords: session files omit them")

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = [f"SessionManager({len(self.entries)} entries):"]
        for i, e in enumerate(self.entries):
            marker = "→" if i == self._active_idx else " "
            lines.append(f"  {marker} [{i}] {e.status:12s} {e.stem}")
        return "\n".join(lines)
