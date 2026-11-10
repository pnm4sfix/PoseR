"""Data models for pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class InferenceMode(str, Enum):
    """How YOLO runs over a video's frames.

    Subclasses str rather than enum.StrEnum, which needs Python 3.11 while
    this package supports 3.10.

    Attributes:
        PREDICT: Every frame on its own, in batches. Fast, but an animal's ID
            is not carried from one frame to the next.
        TRACK: One frame at a time, following each animal across frames so
            its ID stays the same. Needed to tell several animals apart.
    """

    PREDICT = "predict"
    TRACK = "track"


@dataclass(frozen=True, slots=True)
class FrameKeypoints:
    """The animals YOLO found in one video frame.

    P is the number of animals found, 0 when there are none. K is the number
    of body points the model predicts, 19 for zeb.pt.
    """

    frame: int  # index of the frame in the video, from 0
    xy: np.ndarray  # (P, K, 2) float32, x and y in pixels
    conf: np.ndarray  # (P, K) float32, from 0 to 1
    ids: np.ndarray  # (P,) int64, tracker IDs in track mode, else 0 to P-1
