# PoseR

[![License BSD-3](https://img.shields.io/pypi/l/PoseR.svg?color=green)](https://github.com/pnm4sfix/PoseR/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/PoseR-napari.svg?color=green)](https://pypi.org/project/PoseR-napari)
[![Python Version](https://img.shields.io/pypi/pyversions/PoseR-napari.svg?color=green)](https://python.org)
[![tests](https://github.com/pnm4sfix/PoseR/workflows/tests/badge.svg)](https://github.com/pnm4sfix/PoseR/actions)
[![codecov](https://codecov.io/gh/pnm4sfix/PoseR/branch/main/graph/badge.svg)](https://codecov.io/gh/pnm4sfix/PoseR)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/PoseR)](https://napari-hub.org/plugins/PoseR)

A deep learning toolbox for decoding animal behaviour

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->
![alt text](https://github.com/pnm4sfix/PoseR/blob/add-functionality/docs/logo.png?raw=true)

## Installation

### 1. Create a conda environment

    conda create -n PoseR python=3.10
    conda activate PoseR

### 2. Install PyTorch with GPU support

Visit https://pytorch.org/get-started/locally/ and select your OS, CUDA version, and conda/pip.

Example for **CUDA 12.1**:

    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Example for **CPU only**:

    conda install pytorch torchvision torchaudio cpuonly -c pytorch

> **Note:** Install PyTorch *before* PoseR so pip does not overwrite the GPU build with a CPU-only wheel.

### 3. Install PoseR

Latest release from PyPI:

    pip install PoseR-napari

Latest development version from GitHub:

    pip install git+https://github.com/pnm4sfix/PoseR.git

Editable install (for development):

    git clone https://github.com/pnm4sfix/PoseR.git
    cd PoseR
    pip install -e .

### 4. Launch napari

    conda activate PoseR
    napari

Then open the PoseR panels from the **Plugins** menu.


## Quick start

https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/QuickStart.md

## Feature Status

### Data Panel
- [x] Add video files to session
- [x] Add pose files to session (DLC `.h5`, DLC `.csv`, PoseR `.h5`, SLEAP `.h5`)
- [x] Add a folder of matched video + pose file pairs
- [x] **Add zarr array store as video** (multi-camera: splits axis-1 into one layer per camera)
- [x] Activate session entry — loads video and pose layers into napari
- [x] Lazy video loading via pyav + dask (no RAM spike on open)
- [x] Mark entry as done / remove entry
- [x] Navigate to next pending entry
- [x] Save / load session to JSON
- [ ] Progress bar during video load

### Inference Panel
- [x] Pose estimation mode (YOLO-based; bundled zebrafish model + custom `.pt`)
- [x] Run pose estimation on active session entry
- [x] **Zarr video inference** — chunk-aligned decompression, predict and track modes
- [x] Batch pose estimation across all session entries
- [x] Behaviour decoding mode (ST-GCN skeleton-graph model)
- [x] Run behaviour prediction on active session entry
- [x] Batch behaviour prediction across all session entries
- [x] Export predictions (CSV / HDF5)
- [ ] GPU device selector

### Annotation Panel
- [x] Manage behaviour label set (add / edit / remove)
- [x] Navigate detected bouts (prev / next)
- [x] Assign label to selected bout
- [x] Bout-level statistics table
- [x] Export annotations (CSV / HDF5)
- [ ] Auto-populate labels from active session entry on load

### Analysis Panel
- [x] Bout detection — orthogonal movement method
- [x] Bout detection — egocentric movement method
- [x] Per-individual analysis
- [x] Configurable FPS, velocity threshold, and minimum bout length
- [x] Manual bout marking (set start / end frame)
- [x] Bout list with napari frame-jump on selection
- [x] Export frame slices as images
- [ ] Ethogram visualisation (1-D time series viewer)

### Training Panel
- [x] Specify training data files manually
- [x] Use labelled session entries as training data
- [x] Configure model and hyperparameters via YAML config file
- [x] Start / stop training subprocess
- [x] Live training log stream in panel
- [ ] Fine-tune from a pretrained checkpoint
- [ ] Evaluate / test a trained model

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"PoseR" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/pnm4sfix/PoseR/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
