import numpy as np
import h5py
from pathlib import Path


def read_file(file_path):
    assert Path(file_path).exists(), "Please provide valid file path"
    return h5py.File(file_path, 'r')

def write_file(file_path, data):
    # Placeholder for file writing function
    pass
#
# def save_points(points, file_name):
#     # Check if points array is empty
#     if points.size == 0:
#         print(f"Warning: No points to save in {file_name}.")
#         return
#
#     # Handle points with shape (n, 2)
#     if points.shape[1] == 2:
#         tmp = np.zeros((points.shape[0], 3))
#         tmp[:, 0:2] = points
#     else:
#         tmp = points
#
#     with open(file_name, "w") as f:
#         for i in range(tmp.shape[0]):
#             f.write("v {} {} {}\n".format(tmp[i, 0], tmp[i, 1], tmp[i, 2]))
#
# def save_combined_shapes(sampled_shapes, file_name):
#     """
#     Combine and save all shapes from sampled_shapes into a single file.
#
#     Args:
#     sampled_shapes (dict): A dictionary of dictionaries containing numpy arrays of shape points.
#     file_name (str): The name of the file to save the combined points.
#     """
#
#     # Extract all numpy arrays from the nested dictionary
#     all_shapes = []
#     for part_shapes in sampled_shapes.values():
#         for shape in part_shapes.values():
#             if isinstance(shape, np.ndarray) and shape.size > 0:
#                 all_shapes.append(shape)
#
#     # Check if there are valid shapes to combine
#     if not all_shapes:
#         print(f"Warning: No valid points to save in {file_name}.")
#         return
#
#     # Combine valid shapes
#     combined_points = np.vstack(all_shapes)
#
#     # Save using the save_points function
#     save_points(combined_points, file_name)
#
#     return combined_points
#
#
#
#     estimate = estimate_radius(combined_points, target_num_points=1000)
#
#     downsampled_points = downsample_point_cloud(points=combined_points, target_num_points=1000, radius= estimate)
#
#     idx = pcu.downsample_point_cloud_poisson_disk(combined_points, radius=estimate, target_num_samples=1000)
#     pcu_down_sampled_points = combined_points[idx]
#
#     # Save using the save_points function
#     save_points(downsampled_points, "downsampled_points.obj")
#     save_points(pcu_down_sampled_points, "pcu_down_sampled_points.obj")


def extract_and_save_individual_shapes(sampled_shapes, base_file_name):
    """
    Extracts individual shapes from a nested dictionary and saves each to a separate file.

    Args:
        sampled_shapes (dict): A dictionary of dictionaries containing numpy arrays of shape points.
        base_file_name (str): Base name for files to save individual shapes.
    """
    for part_name, part_shapes in sampled_shapes.items():
        for shape_name, shape in part_shapes.items():
            if isinstance(shape, np.ndarray) and shape.size > 0:
                file_name = f"{base_file_name}_{part_name}_{shape_name}.obj"
                save_points_to_file(shape, file_name)


def combine_shapes(sampled_shapes):
    """
    Combine numpy arrays from a nested dictionary structure into a single numpy array.

    Args:
        sampled_shapes (dict): A dictionary of dictionaries containing numpy arrays of shape points.

    Returns:
        np.ndarray: Combined numpy array of all shape points.
    """
    all_shapes = []
    for part_shapes in sampled_shapes.values():
        for shape in part_shapes.values():
            if isinstance(shape, np.ndarray) and shape.size > 0:
                all_shapes.append(shape)

    return np.vstack(all_shapes) if all_shapes else np.array([])

def combine_shapes_with_index(sampled_shapes):
    """
    Combine numpy arrays from a nested dictionary structure into a single numpy array,
    and keep track of the indexes of values for each sub-dictionary key.

    Args:
        sampled_shapes (dict): A dictionary of dictionaries containing numpy arrays of shape points.

    Returns:
        np.ndarray: Combined numpy array of all shape points.
        dict: Dictionary of sub-dictionary keys and their index ranges in the combined array.
    """
    all_shapes = []
    index_ranges = {}
    current_index = 0

    for outer_key, part_shapes in sampled_shapes.items():
        for inner_key, shape in part_shapes.items():
            if isinstance(shape, np.ndarray) and shape.size > 0:
                if current_index > 0:
                    start_index = current_index + 1
                else:
                    start_index = current_index  # Start index for the current inner_key
                all_shapes.append(shape)
                current_index += shape.shape[0]  # Update current index by the number of rows in shape
                end_index = current_index  # End index for the current inner_key (non-inclusive)
                if inner_key not in index_ranges:
                    index_ranges[inner_key] = []
                index_ranges[inner_key].append((start_index, end_index))

    combined_array = np.vstack(all_shapes) if all_shapes else np.array([])
    return combined_array, index_ranges

def find_range(index_ranges, index):
    """
    Finds the range a given index belongs to.

    Args:
        index_ranges (dict): Index ranges for each shape key.
        index (int): The index to locate within the ranges.

    Returns:
        str or None: The key of the range if found, else None.
    """
    for key, ranges in index_ranges.items():
        for start, end in ranges:
            if start <= index < end:
                return key
    return None

def map_points_to_ranges(index_ranges, downsampled_points):
    # Initialize a new dictionary to store the mapped points
    mapped_points = {key: [] for key in index_ranges}

    # Map each point to the correct range
    for index, point in downsampled_points:
        range_key = find_range(index_ranges, index)
        if range_key is not None:
            mapped_points[range_key].append(point)

    return mapped_points


def prepare_points_for_saving(points):
    """
    Prepare points for saving by checking their dimensions and adjusting as necessary.

    Args:
        points (np.ndarray): Numpy array of points to prepare.

    Returns:
        np.ndarray: Adjusted numpy array of points ready for saving.
    """
    if points.size == 0:
        return np.array([])  # Return an empty array if no points

    if points.shape[1] == 2:
        tmp = np.zeros((points.shape[0], 3))
        tmp[:, 0:2] = points
    else:
        tmp = points

    return tmp


def save_points_to_file(points, file_name):
    """
    Save points to a file.

    Args:
        points (np.ndarray): Numpy array of points to save.
        file_name (str): The name of the file where points will be saved.
    """
    if points.size == 0:
        print(f"Warning: No points to save in {file_name}.")
        return

    prepared_points = prepare_points_for_saving(points)

    with open(file_name, "w") as f:
        for i in range(prepared_points.shape[0]):
            f.write("v {} {} {}\n".format(points[i, 0], points[i, 1], points[i, 2]))


def save_combined_shapes(sampled_shapes, file_name):
    """
    Extract, combine, and save all shapes from sampled_shapes into a single file.

    Args:
        sampled_shapes (dict): A dictionary of dictionaries containing numpy arrays of shape points.
        file_name (str): The name of the file to save the combined points.
    """
    combined_points = combine_shapes(sampled_shapes)
    if combined_points.size == 0:
        print(f"Warning: No valid points to save in {file_name}.")
        return

    save_points_to_file(combined_points, file_name)
