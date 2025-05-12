"""
Sampling routines for curves and surfaces.
Provides uniform and random sampling of points along parametric entities.
"""

import numpy as np
from .curve import Curve
from .surface import Surface


def uniform_sample(topo, num_samples, min_pts=None, max_pts=None):
    """
    Sample approximately uniform distributed points on a curve or surface.

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.
    min_pts (int): Minimum number of points to sample.
    max_pts (int): Maximum number of points to sample.


    Returns:
    parametric values and an array of sampled points on the curve/surface.
    """

    if isinstance(topo, Curve):
        return _uniform_sample_curve(topo, num_samples, min_pts, max_pts)
    elif isinstance(topo, Surface):
        return _uniform_sample_surface(topo, num_samples, min_pts, max_pts)
    elif hasattr(topo, 'curve3d') and isinstance(topo.curve3d, Curve):
        return _uniform_sample_curve(topo.curve3d, num_samples, min_pts, max_pts)
    elif hasattr(topo, 'surface') and isinstance(topo.surface, Surface):
        return _uniform_sample_surface(topo.surface, num_samples, min_pts, max_pts)
    else:
        raise ValueError("Invalid geometry type")

def random_sample(topo, num_samples, min_pts=None, max_pts=None):
    """
    Sample random (uniformly random in parameter domain) points on a curve or surface.

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.
    min_pts (int): Minimum number of points to sample.
    max_pts (int): Maximum number of points to sample.


    Returns:
    parametric values and an array of sampled points on the curve/surface.
    """
    if isinstance(topo, Curve):
        return _random_sample_curve(topo, num_samples, min_pts, max_pts)
    elif isinstance(topo, Surface):
        return _random_sample_surface(topo, num_samples, min_pts, max_pts)
    elif hasattr(topo, 'curve3d') and isinstance(topo.curve3d, Curve):
        return _random_sample_curve(topo.curve3d, num_samples, min_pts, max_pts)
    elif hasattr(topo, 'surface') and isinstance(topo.surface, Surface):
        return _random_sample_surface(topo.surface, num_samples, min_pts, max_pts)
    else:
        raise ValueError("Invalid geometry type")



def _uniform_sample_curve(curve, num_samples, min_pts=None, max_pts=None):
    """
    Helper: Uniformly sample points on a Curve.

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the curve.
    """

    # num_samples = max(int(curve.length() / spacing), 1)

    if min_pts is not None:
        num_samples = max(num_samples, min_pts)
    if max_pts is not None:
        num_samples = min(num_samples, max_pts)
    # parameter values evenly spaced over interval
    t = np.linspace(curve.interval[0, 0], curve.interval[0, 1], num_samples).reshape(-1, 1)
    return t, curve.sample(t)


def _random_sample_curve(curve, num_samples, min_pts=None, max_pts=None):
    """
    Helper: Randomly sample points on a Curve.

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the curve.
    """

    if min_pts is not None:
        num_samples = max(num_samples, min_pts)
    if max_pts is not None:
        num_samples = min(num_samples, max_pts)
    # random parameters uniformly distributed in interval
    t = np.random.uniform(low=curve.interval[0, 0], high=curve.interval[0, 1], size=num_samples).reshape(-1, 1)
    return t, curve.sample(t)


def _uniform_sample_surface(surface, num_samples, min_pts=None, max_pts=None):
    """
    Helper: Uniformly sample points on a Surface (grid sampling).

    Args:
    surface: The surface entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the surface.
    """

    if min_pts is not None:
        num_samples = max(num_samples, min_pts)
    if max_pts is not None:
        num_samples = min(num_samples, max_pts)

    num_samples = np.sqrt(num_samples).astype(int)
    # Parameter ranges
    u_values = np.linspace(surface.trim_domain[0, 0], surface.trim_domain[0, 1], num_samples)
    v_values = np.linspace(surface.trim_domain[1, 0], surface.trim_domain[1, 1], num_samples)

    points = np.array(np.meshgrid(u_values, v_values)).T.reshape(-1, 2)
    return points, surface.sample(points)


def _random_sample_surface(surface, num_samples, min_pts=None, max_pts=None):
    """
    Helper: Randomly sample points on a Surface.

    Args:
    surface: The surface entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the surface.
    """

    if min_pts is not None:
        num_samples = max(num_samples, min_pts)
    if max_pts is not None:
        num_samples = min(num_samples, max_pts)
    # Uniform random sampling in the UV domain rectangle
    points = np.random.uniform(low=[surface.trim_domain[0, 0], surface.trim_domain[1, 0]],
                          high=[surface.trim_domain[0, 1], surface.trim_domain[1, 1]],
                          size=(num_samples,2))
    return points, surface.sample(points)

