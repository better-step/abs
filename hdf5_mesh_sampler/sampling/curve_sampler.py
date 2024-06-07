from .sampler import Sampler
import numpy as np



def uniform_sample(curve, spacing):
    """
    Sample uniform points on a curve

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    An array of sampled points on the curve.
    """

    num_samples = max(int(curve.length() / spacing), 1)
    return np.linspace(curve._interval[0], curve._interval[1], num_samples)

def random_sample(curve, spacing):
    """
    Sample random points on a curve

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    An array of sampled points on the curve.
    """

    num_samples = max(int(curve.length() / spacing), 1)
    return np.random.uniform(low=curve._interval[0], high=curve._interval[1], size=num_samples)

def uniform_parametric_sample(curve, spacing):
    """
    Sample uniform points in parametric space on a curve

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    An array of sampled points on the curve.
    """

    num_samples = max(int(abs(curve._interval[1] - curve._interval[0]) / spacing), 1)
    return np.linspace(curve._interval[0], curve._interval[1], num_samples)

def random_parametric_sample(curve, spacing):
    """
    Sample random points in parametric space on a curve

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    An array of sampled points on the curve.
    """

    num_samples = max(int(abs(curve._interval[1] - curve._interval[0]) / spacing), 1)
    return np.random.uniform(low=curve._interval[0], high=curve._interval[1], size=num_samples)
