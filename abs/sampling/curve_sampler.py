import numpy as np


def uniform_sample(curve, spacing):
    """
    Sample uniform points on a curve

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the curve.
    """

    num_samples = max(int(curve.length() / spacing), 1)

    t = np.linspace(curve._interval[0, 0], curve._interval[0, 1], num_samples).reshape(-1, 1)
    return t, curve.sample(t)


def random_sample(curve, spacing):
    """
    Sample random points on a curve

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the curve.
    """

    num_samples = max(int(curve.length() / spacing), 1)

    t = np.random.uniform(low=curve._interval[0, 0], high=curve._interval[0, 1], size=num_samples).reshape(-1, 1)
    return t, curve.sample(t)


def uniform_parametric_sample(curve, spacing):
    """
    Sample uniform points in parametric space on a curve

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the curve.
    """

    num_samples = max(int(abs(curve._interval[0, 1] - curve._interval[0, 0]) / spacing), 1)
    t = np.linspace(curve._interval[0, 0], curve._interval[0, 1], num_samples).reshape(-1, 1)
    return t, curve.sample(t)


def random_parametric_sample(curve, spacing):
    """
    Sample random points in parametric space on a curve

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the curve.
    """

    num_samples = max(int(abs(curve._interval[0, 1] - curve._interval[0, 0]) / spacing), 1)
    t = np.random.uniform(low=curve._interval[0, 0], high=curve._interval[0, 1], size=num_samples).reshape(-1, 1)
    return t, curve.sample(t)
