"""
ABS Geometry Processing Package
"""

"""
ABS geometry-processing toolkit.

Public API
----------
read_parts, read_meshes, sample_parts, poisson_disk_downsample, filter_inside_winding
"""

from importlib import import_module
from warnings import warn
from importlib.metadata import version, PackageNotFoundError

# ── public sub-modules (relative imports!) ────────────────────
from abs.utils import read_parts, read_meshes
from abs.part_processor import sample_parts

# ── optional C++ extension (abspy) ────────────────────────────
try:
    _abspy = import_module("abspy")
    poisson_disk_downsample = _abspy.poisson_disk_downsample
    poisson_grid_downsample = _abspy.poisson_grid_downsample
    filter_inside_winding = _abspy.filter_inside_winding
    BSpline = _abspy.BSpline
    __version__ = version("abs-hdf5")
except ModuleNotFoundError as exc:
    poisson_disk_downsample = None
    poisson_grid_downsample = None
    filter_inside_winding = None
    BSpline = None
    warn(
        "C++ extension 'abspy' not found; blue-noise down-sampling disabled. "
        f"(Original error: {exc})",
        RuntimeWarning,
        stacklevel=2,
    )
except PackageNotFoundError:
    __version__ = "v0.2.0"

__all__ = [
    "read_parts",
    "read_meshes",
    "sample_parts",
    "poisson_disk_downsample",
    "poisson_grid_downsample",
    "filter_inside_winding",
]
