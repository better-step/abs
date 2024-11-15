import numpy as np
from abs.curve import Curve
from abs.surface import Surface
# remove these later
import matplotlib.pyplot as plt
from abs.utils import *



def uniform_sample(geom, num_samples, min_pts=None, max_pts=None):
    """
    Sample uniform points on a curve or surface

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.
    min_pts (int): Minimum number of points to sample.
    max_pts (int): Maximum number of points to sample.


    Returns:
    parametric values and an array of sampled points on the curve/surface.
    """

    if isinstance(geom, Curve):
        return _uniform_sample_curve(geom, num_samples, min_pts, max_pts)
    elif isinstance(geom, Surface):
        return _uniform_sample_surface(geom, num_samples, min_pts, max_pts)
    else:
        raise ValueError("Invalid geometry type")

def random_sample(topo, num_samples, min_pts=None, max_pts=None):
    """
    Sample random points on a curve or surface

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.
    min_pts (int): Minimum number of points to sample.
    max_pts (int): Maximum number of points to sample.


    Returns:
    parametric values and an array of sampled points on the curve/surface.
    """

    if hasattr(topo, '_3dcurve') and isinstance(topo._3dcurve, Curve):
        return _random_sample_curve(topo._3dcurve, num_samples, min_pts, max_pts)
    elif isinstance(topo._surface, Surface):
        return _random_sample_surface(topo._surface, num_samples, min_pts, max_pts)
    else:
        raise ValueError("Invalid geometry type")



#def _uniform_sample_curve(curve, spacing, min_pts=None, max_pts=None):
def _uniform_sample_curve(curve, num_samples, min_pts=None, max_pts=None):
    """
    Sample uniform points on a curve

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

    t = np.linspace(curve._interval[0, 0], curve._interval[0, 1], num_samples).reshape(-1, 1)
    return t, curve.sample(t)


#def _random_sample_curve(curve, spacing, min_pts=None, max_pts=None):
def _random_sample_curve(curve, num_samples, min_pts=None, max_pts=None):
    """
    Sample random points on a curve

    Args:
    curve: The curve entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the curve.
    """

    #num_samples = max(int(curve.length() / spacing), 1)

    if min_pts is not None:
        num_samples = max(num_samples, min_pts)
    if max_pts is not None:
        num_samples = min(num_samples, max_pts)

    t = np.random.uniform(low=curve._interval[0, 0], high=curve._interval[0, 1], size=num_samples).reshape(-1, 1)
    return t, curve.sample(t)


# def _uniform_sample_surface(surface, spacing, min_pts=None, max_pts=None):
def _uniform_sample_surface(surface, num_samples, min_pts=None, max_pts=None):
    """
    Sample uniform points on a surface

    Args:
    surface: The surface entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the surface.
    """

    # num_samples = max(int(np.sqrt(surface.area()) / spacing), 1)

    if min_pts is not None:
        num_samples = max(num_samples, min_pts)
    if max_pts is not None:
        num_samples = min(num_samples, max_pts)

    num_samples = np.sqrt(num_samples).astype(int)

    u_values = np.linspace(surface._trim_domain[0, 0], surface._trim_domain[0, 1], num_samples)
    v_values = np.linspace(surface._trim_domain[1, 0], surface._trim_domain[1, 1], num_samples)

    points = np.array(np.meshgrid(u_values, v_values)).T.reshape(-1, 2)
    return points, surface.sample(points)


def _random_sample_surface(surface, num_samples, min_pts=None, max_pts=None):
    """
    Sample random points on a surface

    Args:
    surface: The surface entity to be sampled.
    spacing (float): The spacing parameter for sampling.


    Returns:
    parametric values and an array of sampled points on the surface.
    """

    # num_samples = max(int(np.sqrt(surface.area()) / spacing), 1)**2

    if min_pts is not None:
        num_samples = max(num_samples, min_pts)
    if max_pts is not None:
        num_samples = min(num_samples, max_pts)

    points = np.random.uniform(low=[surface._trim_domain[0, 0], surface._trim_domain[1, 0]],
                          high=[surface._trim_domain[0, 1], surface._trim_domain[1, 1]],
                          size=(num_samples,2))

    #testing delete later
    # uv_points = points
    # xyz_points = surface.sample(uv_points)
    #
    # plt.figure()
    # plt.scatter(uv_points[:, 0], uv_points[:, 1], alpha=0.5)
    # plt.title("UV Space Distribution")
    # plt.xlabel("U")
    # plt.ylabel("V")
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xyz_points[:, 0], xyz_points[:, 1], xyz_points[:, 2], alpha=0.5)
    # plt.title("3D Space Distribution")
    # plt.show()
    #
    # name = 'sampling'
    # save_file_path = os.path.join(os.path.dirname(__file__), '..', 'test', 'sample_results', f'{name}_normals.obj')
    # save_obj(save_file_path, xyz_points)

    return points, surface.sample(points)

