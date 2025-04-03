# Sampling Module

Provides methods to sample CAD models for feature extraction.

## Key Components

- **Surface Sampler:** Extracts surface samples at specific resolutions.
- **Curve Sampler:** Samples points from curves.

### Usage Example

```python
from hdf5_mesh_sampler.sampling.surface_sampler import sample

points = sample(shape, resolution=0.1)
print("Sampled surface points:", points)
```

> Auto-generated API reference appears here via mkdocstrings.