# Geometry Module

The Geometry module handles the representation and processing of geometric entities in CAD models.

## Key Components

- **Curve Processing:** Classes for lines, circles, ellipses, B-splines, etc.
- **Surface Processing:** Classes for planes, cylinders, cones, spheres, tori, and B-spline surfaces.

### Usage Example

```python
from hdf5_mesh_sampler.geometry.curve import BSplineCurve
import numpy as np

curve_data = {
    'closed': False,
    'degree': 3,
    'poles': [[0, 0], [1, 2], [2, 3], [4, 1]],
    'knots': [0, 0, 0, 1, 2, 2, 2],
    'weights': [1, 1, 1, 1],
    'interval': [[0, 1]],
    'type': 'BSpline'
}

bspline_curve = BSplineCurve(curve_data)
sample_points = np.linspace(0, 1, 100)
points = bspline_curve.sample(sample_points)
print(points)
```

> Auto-generated API reference appears here via mkdocstrings.