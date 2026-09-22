"""Data models for batch processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BatchResult:
    """Outcome of processing one (pose_file, video_file) pair.

    Attributes:
        output_path: Empty when the file failed, and also when the run
            produced nothing to write.
        status: Either "ok" or "error".
        error: Empty unless status is "error".
    """

    pose_path: str
    video_path: str
    output_path: str
    status: str
    error: str = ""
    metadata: Dict = field(default_factory=dict)
