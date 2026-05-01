# Better STEP: A Standardized B-Rep Dataset and Pipeline for Reproducible CAD Evaluation

[![image](https://img.shields.io/pypi/v/abs-hdf5.svg)](https://pypi.python.org/pypi/abs-hdf5)

## ABS-HDF5

ABS-HDF5 reads B-Rep geometry from HDF5 files as a standard half-edge data structure, allowing you to navigate topology (faces, edges, halfedges, loops), sample points directly from continuous parametric surfaces and curves, and evaluate normals and derivatives. It supports user-specified tasks, and provides fast Poisson-disk downsampling for blue-noise point cloud generation.

> HDF5 files are produced by [steptohdf5](https://github.com/better-step/cadmesh), a companion converter that turns STEP/STP CAD files into the format ABS reads.

## Install

```bash
pip install abs-hdf5
```
## Usage

### Reading and traversing topology

`read_parts` loads the full B-Rep structure into memory as linked Python objects.

```python
from abs import read_parts

parts = read_parts('model.hdf5')
face = parts[0].faces[1]

adjacent_faces = face.find_adjacent_faces()
loop = face.loops[0]
```

### Sampling points with user-defined labels

`sample_parts` generates points on parametric surfaces and curves and passes them to your callbacks. Each callback receives the topological entity and the sampled points, and returns whatever per-point values your task needs including labels, normals, primitive types, etc.

```python
from abs import read_parts, sample_parts
import numpy as np

def face_func(face, points):
    return np.zeros(points.shape[0])

def edge_func(edge, points):
    return np.ones(points.shape[0])

parts = read_parts('model.hdf5')
face_points, face_labels, edge_points, edge_labels = sample_parts(
    parts, num_samples=5000, face_func=face_func, edge_func=edge_func
)
```

### Computing normals

Normals are evaluated directly from the underlying parametric surfaces.

```python
def get_normals(face, points):
    return face.normal(points)

face_points, face_normals, _, _ = sample_parts(
    parts, num_samples=5000, face_func=get_normals
)
```

### Extracting the mesh

The mesh is stored independently from the B-Rep topology. `get_mesh` concatenates all per-face triangulations into a single consistent mesh.

```python
from abs.utils import read_meshes, get_mesh

meshes = read_meshes('model.hdf5')
V, F = get_mesh(meshes)
```

## Dataset

The dataset comprises over one million B-Rep models converted from two large-scale public CAD collections:

- **ABC** — ~1,000,000 models
- **Fusion 360 Gallery** — Assembly (162,707 parts), Joint (23,029 parts), Reconstruction (27,958 parts), and Segmentation (35,680 parts)

Each model is stored in HDF5 as a collection of parts, where every part contains its geometry, topology, and an optional triangular mesh. Once converted, no CAD kernel is required to read or process the data.

The dataset can be downloaded here: [link coming soon]

## Converting your own STEP files

To convert your own STEP files, please refer to the [steptohdf5](https://github.com/better-step/cadmesh) repository.
## Contributing

Contributions are welcome. Please open an issue before submitting larger changes.

## License

MIT
