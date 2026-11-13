"""Plain-language help for the Inference panel's options, behind a "?" button."""

from __future__ import annotations

from qtpy.QtWidgets import QMessageBox, QToolButton

OPTION_HELP = {
    "pretrained_model": (
        "Pose model",
        "The YOLO network that finds body parts in each video frame.\n\n"
        "zeb.pt, fly3.pt, mouse7.pt and mouse13.pt are PoseR's own models, "
        "downloaded from the project's GitHub release on first use. The "
        "yolo11*-pose entries are generic COCO human-pose models and will "
        "produce 17 human keypoints on an animal, which is almost never what "
        "you want.\n\n"
        "Sizes run n < s < m < l < x: larger is slower and more accurate.\n\n"
        "Example: zebrafish footage -> zeb.pt.",
    ),
    "max_individuals": (
        "Max individuals",
        "The largest number of animals the detector may find in one frame, "
        "passed to YOLO as max_det.\n\n"
        "Set it to the number actually in the arena. Setting it higher invites "
        "spurious detections; lower silently drops animals.\n\n"
        "Example: one fish in a dish -> 1.",
    ),
    "inference_mode": (
        "Inference mode",
        "predict treats every frame independently. Faster, and fine when "
        "there is one animal, because identity never has to be carried "
        "forward.\n\n"
        "track links detections between consecutive frames so each animal "
        "keeps a stable identity. Needed for multiple animals; slower, and it "
        "must run frames in order.\n\n"
        "Example: one fish -> predict. Two fish you need to tell apart -> "
        "track.",
    ),
    "batch_size": (
        "Batch size",
        "How many frames go through the network at once, in predict mode "
        "only. track is sequential by nature and ignores this.\n\n"
        "Higher is faster until the GPU runs out of memory. Lower it if "
        "inference crashes with an out-of-memory error.\n\n"
        "Example: 16 is a safe start; try 64 on a large GPU.",
    ),
    "image_size": (
        "Image size",
        "The square size each frame is resized to before detection, in "
        "pixels. It must be a multiple of 32.\n\n"
        "It need not match your video. Larger finds small body parts more "
        "reliably and costs time; smaller is faster and may miss them. On "
        "auto, the model's own training size is used.\n\n"
        "Example: a 210x230 video at imgsz 256 is upscaled slightly, which is "
        "fine.",
    ),
    "zarr_camera": (
        "Zarr camera axis",
        "Only relevant for multi-camera zarr arrays, which store every "
        "camera's view in one array. This picks which camera to run on.\n\n"
        "Ordinary video files ignore it.\n\n"
        "Example: a (frames, 4, H, W, 3) array filmed from 4 angles -> 0 for "
        "the first camera.",
    ),
    "checkpoint": (
        "Decoder checkpoint",
        "The trained classifier that turns pose into behaviour labels. A .ckpt "
        "written by training, not a pose model.\n\n"
        "It must match your data: same number of skeleton nodes and the same "
        "input channels. Run 'poser model inspect <file>' to see what a "
        "checkpoint expects before loading it.\n\n"
        "Example: 19-node zebrafish pose -> a decoder reporting num_nodes 19.",
    ),
}


def help_button(key: str) -> QToolButton:
    """A small ? that explains one option, on hover and on click."""
    title, body = OPTION_HELP[key]
    btn = QToolButton()
    btn.setText("?")
    btn.setToolTip(body)
    btn.setAutoRaise(True)
    btn.setFixedWidth(22)
    btn.clicked.connect(lambda: QMessageBox.information(btn, title, body))
    return btn
