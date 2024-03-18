import numpy as np
from scipy.spatial import cKDTree
import point_cloud_utils as pcu


def estimate_radius(points, target_num_points):
    """
    Estimate an appropriate radius for Poisson disk sampling based on the target number of points
    and the spatial extent of the input point cloud.
    """
    bbox_min, bbox_max = np.min(points, axis=0), np.max(points, axis=0)
    bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
    volume_estimate = np.power(bbox_diagonal, 3)  # Assuming a roughly cubic distribution for simplicity
    density_estimate = target_num_points / volume_estimate
    radius = np.cbrt(3 / (4 * np.pi * density_estimate))  # Derive radius from estimated density
    return radius

def generate_random_directions(num_directions=30, dimension=3):
    directions = np.random.randn(num_directions, dimension)
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize to unit vectors
    return directions

def blue_noise_downsample(points, radius, target_num_points):
    if len(points) < target_num_points:
        print("Warning: target_num_points is greater than the number of input points. Returning original points.")
        return points

    kd_tree = cKDTree(points)
    active_list = [np.random.randint(len(points))]
    selected_points_indices = [active_list[0]]

    while active_list and len(selected_points_indices) < target_num_points:
        current_index = active_list.pop(np.random.randint(len(active_list)))
        current_point = points[current_index]
        directions = generate_random_directions()
        found = False

        for direction in directions:
            distance = np.random.uniform(radius, 2 * radius)
            candidate = current_point + direction * distance
            if kd_tree.query(candidate, k=1)[0] >= radius:
                points = np.vstack([points, candidate])
                kd_tree = cKDTree(points)
                selected_points_indices.append(len(points) - 1)
                active_list.append(len(points) - 1)
                found = True
                break

        if not found and not active_list:
            print("Could not find enough points. Try reducing the radius or the target number of points.")
            break

    return points[selected_points_indices]

def voxel_grid_downsample(points, voxel_size):
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    unique_voxel_indices, inverse_indices = np.unique(voxel_indices, return_inverse=True, axis=0)
    downsampled_points = np.array([points[inverse_indices == i].mean(axis=0) for i in range(len(unique_voxel_indices))])
    return downsampled_points

def down_sample_point_cloud(points, target_num_points=None, radius=None, voxel_size=None):
    if voxel_size is not None:
        points = voxel_grid_downsample(points, voxel_size)

    if radius is not None and target_num_points is not None:
        points = blue_noise_downsample(points, radius, target_num_points)
    elif target_num_points is not None and radius is None:
        if len(points) > target_num_points:
            points = points[np.random.choice(len(points), size=target_num_points, replace=False)]
        else:
            print("Warning: target_num_points is greater than the number of input points. Returning original points.")
    return points


def down_sample_point_cloud_pcu(points, target_num_points=None, radius=None):
    """
    Downsample a point cloud using Poisson disk sampling provided by point_cloud_utils, with an optional target number of points or radius.

    Parameters:
    - points: numpy.ndarray of shape (N, 3), the point cloud to downsample.
    - target_num_points: Optional[int], the target number of points. If not provided, radius must be given.
    - radius: Optional[float], the minimum distance between points. If not provided, it's estimated based on the target_num_points.

    Returns:
    - numpy.ndarray of the downsampled point cloud.
    """
    # Check if target_num_points is provided and valid
    if target_num_points is not None and len(points) < target_num_points:
        print("Warning: target_num_points is greater than the number of input points. Returning original points.")
        return points

    # If no radius is provided, estimate it
    if radius is None and target_num_points is not None:
        radius = estimate_radius(points, target_num_points)
    elif radius is None:
        raise ValueError("Either target_num_points or radius must be provided.")

    # Downsample the point cloud
    if target_num_points is not None:
        idx = pcu.downsample_point_cloud_poisson_disk(points, radius=radius, target_num_samples=target_num_points)
        downsampled_points = points[idx]
    else:
        # If no target number of points is provided, downsample using only the radius
        downsampled_points = pcu.downsample_point_cloud_poisson_disk(points, radius=radius)

    return downsampled_points
