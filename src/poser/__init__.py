try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# ---------------------------------------------------------------------------
# Legacy napari widget — preserved for backwards compatibility
# ---------------------------------------------------------------------------
from poser._widget import PoserWidget
from poser._loader import HyperParams, ZebData
from poser.utils import Animation

# ---------------------------------------------------------------------------
# New public API
# ---------------------------------------------------------------------------
from poser.api import PoseR

# Core utilities
from poser.core.io import read_coords, read_dlc, read_sleap, save_to_h5
from poser.core.bout_detection import orthogonal_variance, egocentric_variance, manual_bout
from poser.core.preprocessing import preprocess_bouts
from poser.core.dataset import PoseDataset
from poser.core.session import SessionManager
from poser.core.batch import BatchJob

# Skeleton registry
from poser.skeletons.registry import get_skeleton, list_skeletons

# Model registry
from poser.models.registry import load_model, list_models, register_model

# Training
from poser.training.config import TrainingConfig
from poser.training.trainer import PoseRTrainer

__all__ = [
    # Legacy
    "PoserWidget",
    "HyperParams",
    "ZebData",
    "Animation",
    # New API
    "PoseR",
    "read_coords",
    "read_dlc",
    "read_sleap",
    "save_to_h5",
    "orthogonal_variance",
    "egocentric_variance",
    "manual_bout",
    "preprocess_bouts",
    "PoseDataset",
    "SessionManager",
    "BatchJob",
    "get_skeleton",
    "list_skeletons",
    "load_model",
    "list_models",
    "register_model",
    "TrainingConfig",
    "PoseRTrainer",
]
