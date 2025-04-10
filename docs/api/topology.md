# Topology Module

Describes how geometric entities are connected to form complex 3D structures.

## Key Components

- **Edges:** Boundaries of surfaces tied to 3D curves.
- **Faces:** Surface regions trimmed by loops.
- **Halfedges & Loops:** Fine-grained representation of boundaries.
- **Shells:** Groups of faces forming a boundary.
- **Solids:** Complete enclosed volumes.

### Usage Example

```python
from hdf5_mesh_sampler.topology import Face
import numpy as np

face_data = {...}  # Example dictionary from HDF5
face = Face(face_data)
sample_params = np.linspace(0, 1, 50).reshape(-1,1)
normals = face.normal(sample_params)
print(normals)
```

> Auto-generated API reference appears here via mkdocstrings.