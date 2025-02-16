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

## Installation of PoseR anaconda environment

Download Anaconda-Navigator Desktop application using the following URL: 

    https://www.anaconda.com/download/success

Open up Anaconda-Navigator and launch anaconda_prompt from the home page. Creating the PoseR anaconda environment with the following commands: 

    conda create -n PoseR python=3.10

Activate PoseR environment:

    conda activate PoseR

For the GPU version... Install CUDA IF you are using NVIDIA GPU:

    conda install -c "nvidia/label/cuda-11.7.0" cuda

Install Pytorch IF you are using NVIDIA GPU: 

    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

For CPU only version:

    conda install pytorch torchvision torchaudio cpuonly -c pytorch

Install napari:

    pip install napari[all]==0.4.14 npe2==0.6.2 pydantic==1.10.4


You can install `PoseR` via [pip]:

    pip install PoseR-napari

To install latest issue developed version :

    pip install git+https://github.com/pnm4sfix/PoseR.git


## Quick Start Documentation to utilising PoseR software 

https://github.com/pnm4sfix/PoseR/blob/generalise-species/docs/QuickStart.md

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
