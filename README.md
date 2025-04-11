

# Better Step
[![image](https://img.shields.io/pypi/v/HDF5MeshSampler.svg)](https://pypi.python.org/pypi/HDF5MeshSampler)
[![image](https://img.shields.io/conda/vn/conda-forge/HDF5MeshSampler.svg)](https://anaconda.org/conda-forge/HDF5MeshSampler)

## Description
Better Step is an open‑source project that unlocks CAD data by converting proprietary STEP files into an open, HDF5‑based format. This approach enables efficient processing on large‑scale computing clusters—eliminating expensive per‑node licenses and opening up CAD data for research and industrial applications.
## Key Features
- **Modular Architecture**: The project is structured into distinct modules such as geometry, topology, and sampling, promoting maintainability and scalability.
- **Advanced Geometric Entities**: Supports handling and manipulation of complex geometric shapes, curves, and surfaces.
- **Sophisticated Sampling Techniques**: Implements various sampling methods, with a focus on Poisson disk sampling, tailored for mesh data.
- **Topology Integration**: Seamlessly integrates geometric data with topological structures for comprehensive mesh analysis.
- **HDF5 Data Handling**: Optimized for working with HDF5 file format, ensuring efficient storage and retrieval of large mesh datasets.


## Key Features

- **Open Format:** Converts proprietary CAD files into an accessible HDF5 format.
- **Comprehensive Representation:** Captures both geometry (curves, surfaces) and topology (solids, shells, faces, loops) of CAD models.
- **Robust Sampling Methods:** Provides reliable methods for sampling points, computing normals, detecting sharp features, and generating point clouds for machine learning.
- **Extensible API:** Designed to integrate seamlessly into Python workflows.
- **Command‑Line Interface (CLI):** Supports batch processing and pipeline integration.

## Installation

### Prerequisites

- **Python:** Version 3.7 or later.
- **Dependencies:** HDF5 libraries (via the `h5py` package) and other dependencies listed in `requirements.txt`.

### Via PyPI

Install the package directly from PyPI using pip:

```bash
pip install hdf5_mesh_sampler
```

### From Source

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/better-step.git
   cd better-step
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the Package:**

   ```bash
   python setup.py install
   ```

> **Tip:** It is recommended to use a virtual environment for isolated installations.

## Usage

Better Step provides both a Python API and a command‑line interface for processing CAD data.

### Python API Example

Load a CAD model and sample its geometry:

```python
from hdf5_mesh_sampler import Shape_archive
from hdf5_mesh_sampler.sampling import surface_sampler

# Load an HDF5 file containing CAD data
shape = Shape_archive.load("data/sample_hdf5/Box.hdf5")
print("Loaded shape:", shape)

# Sample the surface with a specified resolution
samples = surface_sampler.sample(shape, resolution=0.1)
print("Sampled points:", samples)
```

### Command‑Line Interface

Process files in batch mode using the CLI:

```bash
python -m hdf5_mesh_sampler.cli --input data/sample_hdf5/Box.hdf5 --output output_directory
```

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
   git clone https://github.com/yourusername/better-step.git
   cd better-step
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

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Better Step is developed and maintained by a dedicated team. For a complete list of contributors, please see the Authors page.
