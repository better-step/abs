# Installation

## Overview

This guide details how to install the Better Step library, available on PyPI as the package `hdf5_mesh_sampler`, or from source via GitHub.

## Prerequisites

- Python 3.7 or later.
- HDF5 libraries (typically installed as part of the `h5py` package).
- Additional dependencies as listed in the project’s requirements file.

## Installation via pip

```bash
pip install hdf5_mesh_sampler
```

## Installation from Source

```bash
git clone https://github.com/yourusername/better-step.git
cd better-step
pip install -r requirements.txt
python setup.py install
```

## Testing Your Installation

```bash
python -c "import hdf5_mesh_sampler; print(hdf5_mesh_sampler.version)"
```

If you see a version number without error, the install is successful.

> Tip: We recommend creating a virtual environment for an isolated installation.
