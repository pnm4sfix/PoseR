# PoseR User Guide

**Version:** 2.0 · **Last updated:** February 2026

PoseR is a multi-species behaviour decoding toolkit built on top of napari.  
It ingests pose estimation keypoints (DeepLabCut, SLEAP, or PoseR-native), detects
locomotion bouts, lets you label them, trains a graph-neural-network classifier
(ST-GCN), and runs inference — all without leaving napari or, if you prefer,
entirely from the command line or a Python script.

---

## Contents

1. [Installation & Setup](#1-installation--setup)
2. [Concepts & Data Flow](#2-concepts--data-flow)
3. **Workflows**
   - [W1 — I already have pose estimation; I want to annotate behaviours](#w1--i-already-have-pose-estimation-i-want-to-annotate-behaviours)
   - [W2 — I need to run pose estimation first, then annotate](#w2--i-need-to-run-pose-estimation-first-then-annotate)
   - [W3 — I want to train a behaviour decoder from my annotations](#w3--i-want-to-train-a-behaviour-decoder-from-my-annotations)
   - [W4 — I want to fine-tune a pose estimation model for my species](#w4--i-want-to-fine-tune-a-pose-estimation-model-for-my-species)
   - [W5 — I want to extract swim bouts and save the raw trajectories](#w5--i-want-to-extract-swim-bouts-and-save-raw-trajectories)
   - [W6 — I want to define behaviours manually (no automatic detection)](#w6--i-want-to-define-behaviours-manually-no-automatic-detection)
   - [W7 — I have pre-trained pose + behaviour models; I want to analyse a dataset](#w7--i-have-pre-trained-pose--behaviour-models-i-want-to-analyse-a-dataset)
   - [W8 — Full pipeline: pose → annotate → train → deploy](#w8--full-pipeline-pose--annotate--train--deploy)
   - [W9 — Active learning: use a weak model to pre-label, then correct](#w9--active-learning-use-a-weak-model-to-pre-label-then-correct)
   - [W10 — Cross-species transfer: adapt a zebrafish model to mouse](#w10--cross-species-transfer-adapt-a-zebrafish-model-to-mouse)
   - [W11 — Batch analysis: process a whole experimental cohort overnight](#w11--batch-analysis-process-a-whole-experimental-cohort-overnight)
   - [W12 — Python / notebook API (no GUI required)](#w12--python--notebook-api-no-gui-required)
4. [CLI Reference](#4-cli-reference)
5. [Configuration Reference](#5-configuration-reference)
6. [Defining Custom Skeletons](#6-defining-custom-skeletons)
7. [Registering Custom Model Architectures](#7-registering-custom-model-architectures)
8. [Suggested Additional Features & Roadmap](#8-suggested-additional-features--roadmap)

---

## 1. Installation & Setup

```bash
# 1. Create a dedicated environment (Python 3.10+)
conda create -n poser python=3.10
conda activate poser

# 2. Install PoseR
pip install poser-napari

# GPU support (optional but recommended for training)
pip install "poser-napari[gpu]"

# 3. Scaffold a new project folder
mkdir my_project && cd my_project
poser init --species zebrafish --classes 4
# Creates decoder_config.yml pre-filled for zebrafish with 4 behaviour classes

# 4. Launch napari
napari
# → Plugins → PoseR
```

**Supported input formats**

| Source | File extension | Notes |
|--------|---------------|-------|
| DeepLabCut | `.h5`, `.csv` | Multi-animal supported |
| SLEAP | `.h5` (Analysis export) | |
| PoseR native | `.h5` | Written by PoseR itself |

---

## 2. Concepts & Data Flow

```
Video file                  Pose file (.h5 / .csv)
     │                             │
     └──────────────┬──────────────┘
                    ▼
            [Data Panel]
        Load into Session table
                    │
                    ▼
          [Analysis Panel]
      Detect bouts  OR  draw manually
                    │
                    ▼
        [Annotation Panel]
         Assign behaviour labels
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   Save  .h5           [Inference Panel]
  (training data)    Predict with checkpoint
                              │
                              ▼
                    predictions.json / .npy
```

**Key terms**

| Term | Meaning |
|------|---------|
| **Session** | A collection of (pose file, video) pairs loaded together. Survives as `session.json`. |
| **Bout** | A time window (start frame → end frame) of candidate behaviour. |
| **Classification file** | An HDF5 file storing egocentric pose clips + their behaviour labels, used as training data. |
| **Skeleton** | A species-specific adjacency graph (edges between keypoints). Built-in or custom YAML. |
| **Checkpoint** | A `.ckpt` or `.pt` file holding trained model weights. |

---

## W1 — I already have pose estimation; I want to annotate behaviours

**You have:** DLC / SLEAP `.h5` files and a matching video.  
**You want:** Labelled behaviour clips ready for training.

### napari GUI

1. `Plugins → PoseR` — the plugin opens with **Data**, **Analysis**, **Annotation**, **Inference**, and **Train** panels docked on the right.

2. **Data Panel → Add Pose File(s)…** — select your `.h5` output file(s).  
   Optionally select a video alongside — it will be loaded as a napari `Image` layer.

3. The file(s) appear in the **Session Table** as ⚪ pending. Double-click (or **Activate**) to load the first one.

4. **Analysis Panel**
   - Set *FPS* to match your recording.
   - Choose method: `orthogonal` (default, works for any locomoting animal) or `egocentric`.
   - Adjust the *Threshold* slider until the detected bouts align with visible movement.
   - Click **▶ Detect Bouts Automatically**.

5. **Annotation Panel** — the bout list populates.
   - Double-click a bout → the viewer seeks to that frame.
   - Click a label button (e.g. `[0] swim`, `[1] freeze`) to assign and auto-advance.
   - Keyboard shortcut: press the digit key matching the label index.

6. When all bouts are labelled: **Export Labels → Save Labelled Bouts (HDF5)…**  
   → saves `classification.h5` in your project folder.

7. Move to the next file: **Data Panel → Next Pending ▶** or double-click the next entry.  
   Repeat steps 4–6 for each file.  
   Mark files as ✅ done when finished.

8. At any point: **Data Panel → Save Session** → `session.json`.  
   Resume another day: **Load Session**.

### Python API

```python
from poser import PoseR

p = PoseR(config="decoder_config.yml")

# Load pose data
coords = p.load("recording_01_DLC.h5")

# Detect bouts
bouts = p.detect_bouts(coords, individual="ind1", fps=25)

# Preprocess into model-ready tensor
X = p.preprocess(coords, bouts, individual="ind1")

# At this point you label X manually, then save
# (see W3 for training once you have labels)
```

---

## W2 — I need to run pose estimation first, then annotate

**You have:** Raw video files only.  
**You want:** Keypoint detections, then behaviour labels.

### Step A — Run YOLOv8 pose estimation

```bash
# From the command line, PoseR wraps Ultralytics YOLO:
poser batch video1.mp4 video2.mp4 \
    --mode pose_estimation \
    --weights yolov8m-pose.pt \
    --individuals 1 \
    --output pose_output/
```
Each video produces a PoseR-native `_coords.h5` file in `pose_output/`.

### Step B — View and annotate

Continue from **W1** using the generated `.h5` files.  
The video file alongside the `.h5` will be detected automatically if named identically.

### napari GUI (combined)

1. **Data Panel → Add Folder of Pose Files…** — point to `pose_output/`.
2. All detected `.h5` files load into the session.
3. Open **Inference Panel → Browse for checkpoint** → select your YOLO `.pt`.
4. Click **▶▶ Run Batch Inference** to detect poses for all files now.
5. Continue to **Analysis Panel** for bout detection → **Annotation Panel** for labelling.

### Python API

```python
from poser import PoseR, BatchJob
from poser.training.config import TrainingConfig

cfg = TrainingConfig.from_yaml("decoder_config.yml")

job = BatchJob(
    pose_files=[],           # empty — video mode
    video_files=["v1.mp4", "v2.mp4"],
    mode="pose_estimation",
    checkpoint="yolov8m-pose.pt",
    config=cfg,
    output_dir="pose_output/",
)
results = job.run()
```

---

## W3 — I want to train a behaviour decoder from my annotations

**You have:** One or more `classification.h5` files produced by W1 or W6.  
**You want:** A trained `.ckpt` model checkpoint.

### napari GUI

1. **Train Panel → Add labelled HDF5 files…** (or **Use session files (done only)**).
2. Optionally browse for a `decoder_config.yml` to change architecture / hyperparameters.
3. Set *Max epochs* and a *Run name*.
4. Click **▶ Start Training**.  
   A live log streams subprocess output into the panel.  
   The napari GUI stays fully responsive throughout.
5. When complete, the best checkpoint path is printed in the log.  
   Checkpoints are saved to `poser_runs/<run_name>/checkpoints/best-*.ckpt`.

### CLI

```bash
# Single command — all defaults from decoder_config.yml
poser train annotations_day1.h5 annotations_day2.h5 \
    --config decoder_config.yml \
    --epochs 200 \
    --name zebrafish_v1
```

### Python API

```python
from poser import PoseR

p = PoseR(config="decoder_config.yml")

# Launches training as a subprocess — GUI / notebook stays responsive
p.train(
    files=["annotations_day1.h5", "annotations_day2.h5"],
    epochs=200,
    run_name="zebrafish_v1",
    blocking=True,   # set False to run in background
)
```

### Tuning the config

Edit `decoder_config.yml` (or use `poser init` to regenerate it):

```yaml
model:
  architecture: st_gcn_3block
  layout: zebrafish          # skeleton name
  num_class: 4
  num_nodes: 9

data:
  fps: 25
  T: 100                     # clip length (frames)
  T2: 100                    # padded length fed to model
  C: 3                       # channels: 2=xy only, 3=xy+confidence

trainer:
  max_epochs: 200
  batch_size: 32
  val_split: 0.2
  early_stopping_patience: 20

optimiser:
  learning_rate: 0.001
  auto_lr: false             # set true to run LR-finder

behaviour_schema:
  labels:
    0: swim
    1: freeze
    2: turn
    3: burst
```

---

## W4 — I want to fine-tune a pose estimation model for my species

**You have:** A pre-trained YOLO pose model (e.g. `yolov8m-pose.pt`) and a small set
of manually annotated images (or frames exported from videos).  
**You want:** A species-specific YOLO model that detects *your* keypoints.

### Step A — Export labelled frames from napari

In **Analysis Panel**:

1. Load your video and navigate to a representative frame using the viewer slider.
2. **Export Frame for Finetuning → Format: YOLO (pose estimation)**.
3. Click **Save Current Frame…** and choose an output directory.  
   This saves the image and a YOLO-format `.txt` label file.
4. Repeat for 50–300 frames covering diverse poses.

### Step B — Fine-tune

```bash
poser finetune exported_frames/ \
    --weights yolov8m-pose.pt \
    --keypoints 9 \
    --epochs 50 \
    --batch 16 \
    --project my_species_pose
```

Or in Python:

```python
from poser.training.finetune_yolo import finetune_yolo

run_dir = finetune_yolo(
    images_dir="exported_frames/",
    base_weights="yolov8m-pose.pt",
    num_keypoints=9,        # must match your skeleton
    epochs=50,
    freeze_backbone=True,   # only fine-tune the head
    project="my_species_pose",
)
print("Best weights:", run_dir / "weights/best.pt")
```

### Tips
- Start with `freeze_backbone=True` (faster, fewer data needed).
- If accuracy plateaus, set `freeze_backbone=False` for a second longer run.
- Export frames from multiple lighting conditions and body orientations.
- Aim for balanced keypoint visibility across frames.

---

## W5 — I want to extract swim bouts and save raw trajectories

**You have:** A DLC / SLEAP `.h5` file for a zebrafish (or other species).  
**You want:** A CSV / HDF5 of bout start/end times and raw egocentric trajectories
for downstream kinematic analysis — *without* running a classifier.

### Python API

```python
from poser.core.io import read_coords, save_coords_to_h5
from poser.core.bout_detection import orthogonal_variance
import numpy as np
import pandas as pd

# Load
coords = read_coords("fish01_DLC.h5", confidence_threshold=0.6)

# Extract bout timings
ind = coords["ind1"]
points = np.stack([ind["x"], ind["y"]], axis=-1)  # (T, V, 2)

bouts, gauss_signal, threshold, euclidean = orthogonal_variance(
    points,
    center_node=5,   # zebrafish swim-bladder
    fps=25,
    n_nodes=9,
)

print(f"Detected {len(bouts)} bouts")

# Save bout table to CSV
df = pd.DataFrame([
    {"start": b["start"], "end": b["end"], "duration_frames": b["end"] - b["start"]}
    for b in bouts
])
df.to_csv("bouts_fish01.csv", index=False)

# Save raw trajectory HDF5 (coords only, no labels)
save_coords_to_h5(coords, video_file="fish01.mp4")
```

### CLI (batch across a folder)

```bash
poser batch *.h5 \
    --mode behaviour \
    --output bout_output/ \
    --individuals 1
```
Each file gets a `bout_output/<stem>/bouts.npy` and a `batch_manifest.csv`.

---

## W6 — I want to define behaviours manually (no automatic detection)

**You have:** A video with pose data loaded in napari.  
**You want:** Full control over exactly which frames constitute each bout —
e.g. for subtle behaviours that automated variance detection misses.

### napari GUI (Manual Bout Definition — Feature 5)

1. Load your pose file and video (**Data Panel**).
2. Open **Analysis Panel → Manual Bout Definition**.
3. Scrub the viewer's frame slider to the first frame of a behaviour.
4. Click **Set Start**.  
   The panel shows `Range: <frame> to —`.
5. Scrub to the last frame of the behaviour.
6. Click **Set End → Add Bout**.  
   The bout is appended to the detected bout list.
7. Repeat for every behaviour instance.
8. Switch to **Annotation Panel** — your manually drawn bouts appear in the list.
9. Assign labels and export to HDF5 as normal.

### Python API

```python
from poser.core.bout_detection import manual_bout
from poser.core.io import read_coords

coords = read_coords("recording.h5")

# Define bouts by hand — start and end are frame indices
bouts = [
    manual_bout(start=120,  end=250,  coords_data=coords, individual_key="ind1"),
    manual_bout(start=480,  end=610,  coords_data=coords, individual_key="ind1"),
    manual_bout(start=1050, end=1180, coords_data=coords, individual_key="ind1"),
]

# Assign labels
for i, b in enumerate(bouts):
    b["label"] = [0, 1, 0][i]   # swim, freeze, swim

# Preprocess and save
from poser.core.io import save_to_h5
save_to_h5(
    classification_data=bouts,
    video_file="recording.mp4",
    n_nodes=9,
    behaviour_schema={0: "swim", 1: "freeze"},
)
```

---

## W7 — I have pre-trained pose + behaviour models; I want to analyse a dataset

**You have:** A batch of videos, a YOLO pose `.pt`, and a behaviour decoder `.ckpt`.  
**You want:** Behaviour predictions for every individual in every video, with minimal effort.

### One command (CLI)

```bash
poser batch videos/*.mp4 \
    --mode behaviour \
    --checkpoint my_decoder.ckpt \
    --individuals 2 \
    --output results/
```

Output per video:

```
results/
  video1/
    coords.h5          ← pose keypoints
    predictions.npy    ← per-bout predicted labels
    bouts.csv          ← start / end / label / confidence
  ...
  batch_manifest.csv   ← summary table (file, n_bouts, status)
```

### napari GUI

1. **Data Panel → Add Folder of Pose Files…** (or raw videos if using YOLO first).
2. **Inference Panel → Browse for checkpoint** → select your `.ckpt`.
3. **▶▶ Run Batch Inference** — a progress bar tracks each file.
4. Results stream into the log; predictions are saved automatically.

### Python API

```python
from poser import PoseR

p = PoseR(config="decoder_config.yml")

results = p.batch(
    pose_files=["fish01.h5", "fish02.h5", "fish03.h5"],
    checkpoint="zebrafish_decoder_v2.ckpt",
    output_dir="results/",
)

for r in results:
    if r.success:
        print(r.pose_file.stem, r.predictions)
```

---

## W8 — Full pipeline: pose → annotate → train → deploy

This is the complete end-to-end workflow for a new species or context.

```
Day 1: Collect raw videos
         │
         ▼
poser batch videos/ --mode pose_estimation --weights yolov8m-pose.pt
         │
         ▼
Day 2: Review and annotate in napari (W1)
  · Load pose files into Session
  · Detect bouts, label 200–500 examples across conditions
  · Save classification.h5 files
         │
         ▼
Day 3: Train decoder (W3)
poser train annotations/*.h5 --epochs 200 --name v1
         │
         ▼
Day 4: Evaluate
poser predict new_recording.h5 --checkpoint runs/v1/best.ckpt
         │
         ├── Good enough → deploy to full cohort (W7)
         │
         └── Needs more data → add annotations → retrain (W9)
```

### Iterative refinement loop

```bash
# Iteration 1: train on 200 examples
poser train batch1.h5 --name v1 --epochs 100

# Inspect errors
poser predict validation.h5 --checkpoint runs/v1/best.ckpt

# Add 100 more hard examples in napari, save batch2.h5
# Iteration 2
poser train batch1.h5 batch2.h5 --name v2 --epochs 150
```

---

## W9 — Active learning: use a weak model to pre-label, then correct

**Goal:** Minimise manual annotation time by letting an initial model pre-label new data
and only correcting its mistakes.

### Python API

```python
from poser import PoseR
from poser.core.io import read_coords, save_to_h5

p = PoseR(config="decoder_config.yml")

# 1. Load new recording
coords = p.load("new_fish.h5")
bouts   = p.detect_bouts(coords, individual="ind1")

# 2. Pre-label with weak model
predictions = p.predict(
    coords_data=coords,
    checkpoint="weak_model_v1.ckpt",
    individual="ind1",
)
# predictions = {start_frame: predicted_label, ...}

# 3. Attach predictions to bouts for review
for b in bouts:
    b["label"] = predictions.get(b["start"], -1)

# 4. Save pre-labelled bouts → open in napari to correct
save_to_h5(bouts, video_file="new_fish.mp4", n_nodes=9,
           behaviour_schema={0: "swim", 1: "freeze"})

# 5. Open in napari: Annotation Panel shows pre-filled labels
#    Correct any errors, re-save, add to training set
```

### napari GUI flow

1. Run inference on a new file (**Inference Panel**).
2. The predictions populate the **Annotation Panel** alongside the bout list.
3. Cycle through bouts, correct mis-labelled ones, then re-export.
4. Add the corrected file to your next training run.

---

## W10 — Cross-species transfer: adapt a zebrafish model to mouse

ST-GCN learns motion patterns that are partly species-agnostic.  
A zebrafish model fine-tuned on a small amount of mouse data often outperforms
training from scratch.

### Define the mouse skeleton first

```bash
poser skeleton validate mouse1.yaml
poser skeleton info mouse1
```

### Transfer training

```yaml
# decoder_config.yml for mouse
model:
  architecture: st_gcn_3block
  layout: mouse1
  num_class: 3
  num_nodes: 13

trainer:
  max_epochs: 100
  batch_size: 16
```

```bash
# Fine-tune from zebrafish checkpoint
poser train mouse_annotations.h5 \
    --config mouse_config.yml \
    --name mouse_v1
# (Checkpoint initialisation from a zebrafish .ckpt is on the roadmap — see §8)
```

---

## W11 — Batch analysis: process a whole experimental cohort overnight

```bash
# 1. Prepare a file list
Get-ChildItem D:\experiment\*.h5 | Select-Object -ExpandProperty FullName > filelist.txt

# 2. Run overnight
poser batch (Get-Content filelist.txt) `
    --checkpoint models/zebrafish_v3.ckpt `
    --config decoder_config.yml `
    --output results/ `
    --individuals 1

# 3. Next morning — summary in batch_manifest.csv
```

Results directory layout:

```
results/
  fish001_trial1/
    predictions.npy
    bouts.csv
  fish001_trial2/
    ...
  batch_manifest.csv        ← one row per file: status, n_bouts, error
```

### Python (parallel workers)

```python
from poser.core.batch import BatchJob
from poser.training.config import TrainingConfig
from pathlib import Path

cfg  = TrainingConfig.from_yaml("decoder_config.yml")
files = list(Path("data/").glob("*.h5"))

job = BatchJob(
    pose_files=files,
    mode="behaviour",
    checkpoint=Path("models/zebrafish_v3.ckpt"),
    config=cfg,
    output_dir=Path("results/"),
    n_individuals=1,
    progress_callback=lambda d, t, p: print(f"[{d}/{t}] {p}"),
)
results = job.run()
```

---

## W12 — Python / notebook API (no GUI required)

PoseR works entirely without napari for scripting and HPC use.

```python
from poser import PoseR, list_skeletons, get_skeleton
from poser.training.config import TrainingConfig

# Inspect available skeletons
print(list_skeletons())
# ['coco', 'drosophila', 'mouse1', 'mouse2', 'ntu_rgbd', 'openpose',
#  'zeb60fps', 'zebrafish', 'zebrafishlarvae']

skel = get_skeleton("zebrafish")
print(skel.node_names)

# Full pipeline in ~10 lines
p = PoseR(config="decoder_config.yml")

coords   = p.load("fish01_DLC.h5")
bouts    = p.detect_bouts(coords, method="orthogonal")
X        = p.preprocess(coords, bouts)

# (assume you already have a trained model)
preds    = p.predict(coords_data=coords, checkpoint="v3.ckpt")
print(preds)   # {frame: label, ...}
```

### Jupyter notebook tip

```python
# Non-blocking training in a notebook kernel
proc = p.train(
    files=["labelled.h5"],
    epochs=200,
    blocking=False,   # returns immediately
)

# Do other analysis while training runs
import time
while proc.poll() is None:
    print("still training…")
    time.sleep(30)
print("Done, exit code:", proc.returncode)
```

---

## 4. CLI Reference

```
poser --help

Commands:
  train        Train a behaviour decoder from labelled pose files.
  predict      Run inference on a single pose file.
  batch        Batch process multiple files (pose estimation or behaviour).
  finetune     Fine-tune a YOLOv8 pose model on exported frames.
  skeleton     Manage skeleton definitions.
    list         List all built-in and registered skeletons.
    info NAME    Show node names, edges, and centre/head for a skeleton.
    validate     Validate a custom skeleton YAML file.
  model        Manage model architectures.
    list         List registered architectures and discovered checkpoints.
  init         Scaffold a new project folder with a starter config.
```

**Common options**

| Flag | Default | Meaning |
|------|---------|---------|
| `--config` / `-c` | `decoder_config.yml` | Training configuration YAML |
| `--checkpoint` / `-w` | — | Model weights (`.ckpt` or `.pt`) |
| `--output` / `-o` | `batch_output/` | Output directory |
| `--epochs` / `-e` | from config | Override `trainer.max_epochs` |
| `--name` / `-n` | `"run"` | Run sub-directory name |
| `--verbose` / `-v` | off | Enable DEBUG logging |

---

## 5. Configuration Reference

`decoder_config.yml` is the single source of truth for a PoseR experiment.
Generate a starter file with `poser init`, or load programmatically with
`TrainingConfig.from_yaml("decoder_config.yml")`.

```yaml
species: zebrafish                # informational tag

model:
  architecture: st_gcn_3block     # registered arch name
  layout: zebrafish               # skeleton registry key
  num_class: 4                    # number of behaviour classes
  num_nodes: 9                    # must match skeleton
  in_channels: 3                  # 2=xy, 3=xy+confidence
  dropout: 0.5
  edge_importance_weighting: true

data:
  fps: 25
  T: 100                          # input clip length (frames)
  T2: 100                         # padded clip length
  C: 3
  M: 1                            # max individuals per frame
  denominator: 100.0              # coordinate normalisation divisor
  confidence_threshold: 0.6
  center_data: true
  align_data: true
  T_method: pad                   # "pad" or "interpolate"

trainer:
  max_epochs: 200
  batch_size: 32
  val_split: 0.2                  # fraction held out for validation
  early_stopping_patience: 20
  gradient_clip_val: 1.0
  precision: "32"                 # "16-mixed" for half-precision 
  seed: 42
  class_weights: null             # e.g. [1.0, 2.5, 1.0, 1.0]

optimiser:
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler: cosine               # "cosine", "step", or "none"
  warmup_epochs: 5
  auto_lr: false

augmentation:
  rotate: true
  rotate_max_deg: 45.0
  jitter: true
  jitter_sigma: 0.02
  scale: true
  scale_range: [0.8, 1.2]
  roll: true
  fragment: true

behaviour_schema:
  labels:
    0: swim
    1: freeze
    2: turn
    3: burst

output_dir: poser_runs
run_name: zebrafish_v1
```

---

## 6. Defining Custom Skeletons

Copy `src/poser/skeletons/TEMPLATE.yaml` and fill in your keypoints:

```yaml
name: my_fish                # unique name, lowercase
num_nodes: 7

node_names:
  - eye
  - head
  - pectoral
  - dorsal_fin
  - body_mid
  - peduncle
  - tail_tip

# 0-indexed [child, parent] pairs
edges:
  - [0, 1]
  - [1, 2]
  - [1, 3]
  - [3, 4]
  - [4, 5]
  - [5, 6]

center_node: 4
head_node: 0
partition_strategy: spatial   # "spatial", "distance", or "uniform"
```

**Validate it:**

```bash
poser skeleton validate my_fish.yaml
# ✓ Valid skeleton: my_fish  (7 nodes)
```

**Use it in your config:**

```yaml
model:
  layout: my_fish
  num_nodes: 7
```

**Register via package entry-point** (for distribution):

```ini
# setup.cfg / pyproject.toml
[options.entry_points]
poser.skeletons =
    my_fish = my_package.skeletons:MyFishSpec
```

---

## 7. Registering Custom Model Architectures

Sub-class `BasePoseModel` and decorate with `@register_model`:

```python
# my_package/my_arch.py
import torch
from poser.models.base import BasePoseModel
from poser.models.registry import register_model

@register_model("my_transformer", description="Pose Transformer encoder")
class PoseTransformer(BasePoseModel):
    MODEL_NAME    = "my_transformer"
    MODEL_VERSION = "1.0"

    def __init__(self, num_class=2, num_nodes=9, in_channels=3, **kwargs):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoder(...)
        self.head    = torch.nn.Linear(256, num_class)

    def forward(self, x):
        # x: (N, C, T, V, M)
        ...
        return self.head(features)

    def training_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        ...
```

Register via entry-point so PoseR discovers it automatically:

```ini
[options.entry_points]
poser.models =
    my_transformer = my_package.my_arch:PoseTransformer
```

Use in config:

```yaml
model:
  architecture: my_transformer
```

---

## 8. Suggested Additional Features & Roadmap

Based on the workflows above, the following features would significantly strengthen
the platform. They are presented in rough priority order.

### 🔴 High priority (missing functionality gaps)

| # | Feature | Motivation |
|---|---------|-----------|
| 8.1 | **Checkpoint warm-start / transfer learning** | Load pretrained zebrafish weights into a new-species model, freezing earlier graph conv layers. Currently each training run starts from scratch. Required for W10 to work efficiently. |
| 8.2 | **Confidence-based prediction filtering** | Add a `min_confidence` threshold to inference output. Only output labels where the softmax probability exceeds the threshold; flag the rest as `uncertain`. Crucial for deployment. |
| 8.3 | **Napari overlay of predicted labels on the video** | After inference, display colour-coded behaviour label as a `Text` or `Shapes` layer overlaid on the video. Currently predictions are only saved to file. |
| 8.4 | **Bout detection parameter auto-tuning** | Given a known bout (manually drawn), optimise the threshold / FPS window to detect similar bouts in the same recording. Save per-recording parameters to the session. |
| 8.5 | **Multi-individual synchronised display** | When multiple individuals are in a video, show all `Points` layers simultaneously with per-individual colour coding, and detect bouts per-individual in one pass. |

### 🟡 Medium priority (workflow polish)

| # | Feature | Motivation |
|---|---------|-----------|
| 8.6 | **Automated report generation** | After analysis, generate a PDF / HTML report with per-individual bout histograms, prediction confusion matrix, and trajectory plots. `poser report results/` |
| 8.7 | **Semi-supervised annotation assist** | After labelling ~50 examples, train a quick weak model and surface its predictions alongside the bout list in napari so the user can accept or reject. |
| 8.8 | **NWB / BIDS export** | Export pose and behaviour labels in Neurodata Without Borders format for archiving and inter-operability with electrophysiology pipelines. |
| 8.9 | **Experiment tracking integration** | Optional `wandb` / `mlflow` logger passed to `PoseRTrainer`. One line in config: `logging: wandb`. |
| 8.10 | **Skeleton visualiser in napari** | A dedicated panel tab that renders the graph adjacency as a node-link diagram so users can visually verify their custom skeleton before running. |

### 🟢 Lower priority (nice-to-have)

| # | Feature | Motivation |
|---|---------|-----------|
| 8.11 | **Real-time inference during video playback** | Stream video frames → YOLO pose → behaviour decoder, display predicted label live. Useful for behavioural screens. |
| 8.12 | **Data augmentation preview widget** | Show the effect of each augmentation (rotate, jitter, scale) on a sample bout in the napari canvas before committing to a training run. |
| 8.13 | **Cross-validation training mode** | `poser train --cv 5` to run k-fold CV and report mean ± std accuracy. |
| 8.14 | **Behaviour ethogram export** | Timeline plot (individual × frame, coloured by predicted behaviour) exportable as SVG / PNG. |
| 8.15 | **REST API / headless server mode** | `poser serve --port 8080` exposes predict and batch endpoints over HTTP for HPC cluster integration. |

### Implementation notes for 8.1 (warm-start) — recommended next step

```python
# Proposed API addition to PoseRTrainer / api.py
p.train(
    files=["mouse_annotations.h5"],
    pretrained_checkpoint="zebrafish_v3.ckpt",
    freeze_layers=["st_gcn.0", "st_gcn.1"],  # freeze first 2 graph blocks
    epochs=100,
    run_name="mouse_transfer_v1",
)
```

Implementation: load checkpoint state-dict, skip head weights (mismatched `num_class`),
copy backbone weights, set `requires_grad=False` for specified layer names before fitting.  
Estimated complexity: ~30 lines in `training/trainer.py`.

### Implementation notes for 8.3 (label overlay)

After batch inference, create a napari `Labels` or coloured `Shapes` layer whose
segments span `[start, end]` in the time dimension, coloured by predicted class.
Map `behaviour_schema` colours to a fixed palette.  
Estimated complexity: ~50 lines in `_panels/inference_panel.py`.
