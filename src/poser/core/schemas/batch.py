"""Data models for batch processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class BatchMode(str, Enum):
    """What a batch run does to each input file.

    Subclasses str rather than enum.StrEnum, which needs Python 3.11 while
    this package supports 3.10.

    Attributes:
        BEHAVIOUR: Detect bouts in a pose file and classify them.
        POSE_ESTIMATION: Run YOLO-pose over a video to produce a pose file.
    """

    BEHAVIOUR = "behaviour"
    POSE_ESTIMATION = "pose_estimation"


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
