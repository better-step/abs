

# ABS-HDF5: Geometry processing & blue-noise sampling for HDF5 B-Rep data
[![image](https://img.shields.io/pypi/v/abs-hdf5.svg)](https://pypi.python.org/pypi/abs-hdf5)


## Description
ABS-HDF5 is a Python + C++ toolkit that turns the raw B-Rep information stored in
**HDF5** CAD datasets into immediately useful geometry:

* read **curves, surfaces, topology** into convenient Python objects
* sample points on faces/edges with exact parametric control
* perform fast **Poisson-disk (blue-noise) down-sampling** via a native
  pybind11 C++ extension
* export ready-to-visualise point clouds in **PLY** or analysis-ready
  **Pickle** files
---

## Quick install

```bash
pip install abs-hdf5        # pre-built wheels for CPython 3.8 – 3.12
```

Building from source?  You need

* Python 3.8 +
* a C++17 compiler (GCC 9 / Clang 12 / MSVC 17.6 or newer)
* CMake ≥ 3.22 and Ninja

```bash
pip install .  # inside a git checkout – scikit-build-core will compile abspy
```

---

## Command-line tools

### `abs-to-ply`

Convert one file (or a directory of *.hdf5* files) to point-cloud PLY.

```bash
abs-to-ply data/Cone.hdf5   out/          -n 5000  -j 8
# ^input file / dir          ^output dir   ^pts/part ^workers
```

Each face gets uniformly-random samples; Poisson-disk down-sampling keeps only
*5000* nearly-evenly-spaced points per part.
Normals are written alongside every vertex in ASCII PLY.

### `abs-to-pickle`

Same interface, but saves a per-part `*.pkl` with a dict:

```python
{'file': 'Cone.hdf5', 'part': 0, 'points': ndarray(N,3), 'normals': ndarray(N,3)}
```

Useful for machine-learning pipelines that prefer NumPy-pickles.

---

## Library use

```python
import abs                       # auto-loads the C++ extension `abspy`

# Read every part stored in a single HDF5 file
parts = abs.read_parts("data/Cylinder.hdf5")

# How to extract 10 000 blue-noise samples on each part (with normals)
def face_normals(part, topo, uv):
    return topo.normal(uv) if topo.is_face() else None

points_per_part, normals_per_part = abs.sample_parts(
        parts, num_samples=10_000, lambda_func=face_normals)

# Poisson-disk down-sample an arbitrary XYZ array to 1 000 pts
idx = abs.poisson_disk_downsample(points_per_part[0], 1000)
sub_pts = points_per_part[0][idx]
```

See the doc-strings in `abs.part_processor`, `abs.sampler`, and
`abs.shape.Shape` for the full low-level API.

> **Tip:** It is recommended to use a virtual environment for isolated installations.


For more usage details, please refer to the Usage page in our documentation.

## Documentation

Detailed documentation is available on our website. It includes:

- **Installation Guide**
- **Usage Examples**
- **API Documentation:** Covers modules like Geometry, Topology, Sampling, and Visualization.
- **FAQ, Changelog, Contributing Guidelines, and More**

Visit our website: [better-step.github.io](https://better-step.github.io)

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository:**
   Click the "Fork" button on GitHub.

2. **Clone Your Fork Locally:**

   ```bash
   git clone https://github.com/better-step/abs.git
   cd abs
   python -m pip install -e .
   pytest
   ```

3. **Create a Branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes:**
   Ensure your code follows our coding standards (PEP 8) and include docstrings and tests where applicable.

5. **Commit and Push:**

   ```bash
   git add .
   git commit -m "Add feature: [description]"
   git push origin feature/your-feature-name
   ```

6. **Submit a Pull Request:**
   Open a pull request on GitHub with a clear description of your changes.

> **Note:** Please also review our Code of Conduct before contributing.

## License


- **Python bindings, packaging scripts, tests** – **MIT License** (see `LICENSE-MIT`)
- **Embedded C++ core** – **Mozilla Public License 2.0** (see `LICENSE-MPL-2.0`)

## Acknowledgments

Better Step is developed and maintained by a dedicated team. For a complete list of contributors, please see the Authors page.

---

*Happy sampling!*  – The Better Step maintainers
