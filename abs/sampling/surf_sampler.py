import numpy as np


def uniform_sample(surface, spacing):
    """
    Sample uniform points on a surface

    Args:
    surface: The surface entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the surface.
    """

    num_samples = max(int(np.sqrt(surface.area()) / spacing), 1)

    u_values = np.linspace(surface._trim_domain[0, 0], surface._trim_domain[0, 1], num_samples)
    v_values = np.linspace(surface._trim_domain[1, 0], surface._trim_domainl[1, 1], num_samples)

    points = np.array(np.meshgrid(u_values, v_values)).T.reshape(-1, 2)
    return points, surface.sample(points)


def random_sample(surface, spacing):
    """
    Sample random points on a surface

    Args:
    surface: The surface entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the surface.
    """

    num_samples = max(int(np.sqrt(surface._area) / spacing), 1)

    u_values = np.random.uniform(low=surface._trim_domain[0, 0], high=surface._trim_domain[0, 1], size=num_samples)
    v_values = np.random.uniform(low=surface._trim_domain[1, 0], high=surface._trim_domain[1, 1], size=num_samples)

    points = np.array(np.meshgrid(u_values, v_values)).T.reshape(-1, 2)
    return points, surface.sample(points)


def uniform_parametric_sample(surface, spacing):
    """
    Sample uniform points in parametric space on a surface

    Args:
    surface: The surface entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the surface.
    """

    num_samples = max(int(np.sqrt(surface.area()) / spacing), 1)

    u_values = np.linspace(surface._trim_domain[0, 0], surface._trim_domain[0, 1], num_samples)
    v_values = np.linspace(surface._trim_domain[1, 0], surface._trim_domain[1, 1], num_samples)

    return np.array(np.meshgrid(u_values, v_values)).T.reshape(-1, 2)


def random_parametric_sample(surface, spacing):
    """
    Sample random points in parametric space on a surface

    Args:
    surface: The surface entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the surface.
    """

    num_samples = max(int(np.sqrt(surface.area()) / spacing), 1)

    u_values = np.random.uniform(low=surface._trim_domain[0, 0], high=surface._trim_domain[0, 1], size=num_samples)
    v_values = np.random.uniform(low=surface._trim_domain[1, 0], high=surface._trim_domain[1, 1], size=num_samples)

    return np.array(np.meshgrid(u_values, v_values)).T.reshape(-1, 2)

