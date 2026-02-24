"""PoseR core layer — pure, napari-free data processing."""

from .augmentation import (
    rotate_transform,
    jitter_transform,
    scale_transform,
    shear_transform,
    roll_transform,
    fragment_transform,
    random_augmentation,
)
from .io import (
    read_dlc,
    read_sleap,
    read_poser_coords,
    read_coords,
    read_classification_h5,
    save_to_h5,
    save_coords_to_h5,
    convert_dlc_to_ctvm,
)
from .bout_detection import (
    orthogonal_variance,
    egocentric_variance,
    manual_bout,
    check_behaviour_confidence,
)
from .preprocessing import (
    preprocess_bouts,
    classification_data_to_bouts,
)
from .dataset import PoseDataset
from .session import SessionManager, SessionEntry
from .batch import BatchJob
from .frame_export import save_frame_as_yolo, save_frame_as_classification
from .metrics import benchmark_model_performance

__all__ = [
    "rotate_transform", "jitter_transform", "scale_transform",
    "shear_transform", "roll_transform", "fragment_transform",
    "random_augmentation",
    "read_dlc", "read_sleap", "read_poser_coords", "read_coords",
    "read_classification_h5", "save_to_h5", "save_coords_to_h5",
    "convert_dlc_to_ctvm",
    "orthogonal_variance", "egocentric_variance", "manual_bout",
    "check_behaviour_confidence",
    "preprocess_bouts", "classification_data_to_bouts",
    "PoseDataset",
    "SessionManager", "SessionEntry",
    "BatchJob",
    "save_frame_as_yolo", "save_frame_as_classification",
    "benchmark_model_performance",
]
