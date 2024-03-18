import numpy as np
import h5py
from pathlib import Path
import point_cloud_utils as pcu
from scipy.spatial import cKDTree
import igl



def read_file(file_path):
    assert Path(file_path).exists(), "Please provide valid file path"
    return h5py.File(file_path, 'r')

def write_file(file_path, data):
    # Placeholder for file writing function
    pass


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

def save_points(points, file_name):
    # Check if points array is empty
    if points.size == 0:
        print(f"Warning: No points to save in {file_name}.")
        return

    # Handle points with shape (n, 2)
    if points.shape[1] == 2:
        tmp = np.zeros((points.shape[0], 3))
        tmp[:, 0:2] = points
    else:
        tmp = points

    with open(file_name, "w") as f:
        for i in range(tmp.shape[0]):
            f.write("v {} {} {}\n".format(tmp[i, 0], tmp[i, 1], tmp[i, 2]))

def save_combined_shapes(sampled_shapes, file_name):
    """
    Combine and save all shapes from sampled_shapes into a single file.

    Args:
    sampled_shapes (dict): A dictionary of dictionaries containing numpy arrays of shape points.
    file_name (str): The name of the file to save the combined points.
    """

    # Extract all numpy arrays from the nested dictionary
    all_shapes = []
    for part_shapes in sampled_shapes.values():
        for shape in part_shapes.values():
            if isinstance(shape, np.ndarray) and shape.size > 0:
                all_shapes.append(shape)

    # Check if there are valid shapes to combine
    if not all_shapes:
        print(f"Warning: No valid points to save in {file_name}.")
        return

    # Combine valid shapes
    combined_points = np.vstack(all_shapes)

    # Save using the save_points function
    save_points(combined_points, file_name)



    estimate = estimate_radius(combined_points, target_num_points=1000)

    downsampled_points = downsample_point_cloud(points=combined_points, target_num_points=1000, radius= estimate)

    idx = pcu.downsample_point_cloud_poisson_disk(combined_points, radius=estimate, target_num_samples=1000)
    pcu_down_sampled_points = combined_points[idx]

    # Save using the save_points function
    save_points(downsampled_points, "downsampled_points.obj")
    save_points(pcu_down_sampled_points, "pcu_down_sampled_points.obj")


# use downsample the point cloud to a specified number of points







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

def downsample_point_cloud(points, target_num_points=None, radius=None, voxel_size=None):
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
