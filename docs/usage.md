# Usage

## Overview

Learn how to use the Better Step library to process CAD files. Load shapes from HDF5 files, sample geometry, and utilize the CLI for batch operations.

## Basic Workflow

### Loading a CAD File

```python
from hdf5_mesh_sampler import Shape_archive
shape = Shape_archive.load("data/sample_hdf5/Box.hdf5")
print("Loaded shape:", shape)
```

### Sampling Geometry

```python
from hdf5_mesh_sampler.sampling import surface_sampler
samples = surface_sampler.sample(shape, resolution=0.1)
print("Sampled points:", samples)
```

### Using the CLI

```bash
python -m hdf5_mesh_sampler.cli --input data/sample_hdf5/Box.hdf5 --output output_directory
```

## Advanced Use Cases

- **Learning Normals:** Use sample data to train or validate normal estimation.
- **Detecting Sharp Features:** Analyze curves to detect geometric edges.
- **Labeling Geometry:** Assign labels based on detected primitive types.

> See the example notebooks in the Examples section for more details.
