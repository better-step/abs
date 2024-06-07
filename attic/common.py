"""The common module contains common functions and classes used by the other modules.
"""

import os
import numpy as np

def get_file_paths(source_path):
    """List all HDF5 files in the source directory."""
    return [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.hdf5')]

def split_dataset(file_paths, split_ratios):
    """Split the dataset into train, validation, and test sets based on the given ratios."""
    # Example implementation, adjust as necessary
    total_files = len(file_paths)
    train_end = int(total_files * split_ratios['train'])
    validation_end = train_end + int(total_files * split_ratios['validation'])

    train_files = file_paths[:train_end]
    validation_files = file_paths[train_end:validation_end]
    test_files = file_paths[validation_end:]

    return train_files, validation_files, test_files


def find_indices_of_points(combined_points, selected_points):
    """
    Find indices of selected points in the combined points array.

    Args:
        combined_points (np.ndarray): The combined numpy array of all shape points.
        selected_points (np.ndarray): The array of selected points.

    Returns:
        np.ndarray: Indices of selected points in the combined points array.
    """
    indices = []
    for point in selected_points:
        # Convert point to tuple to use it as an element in the search
        # This assumes that all points are unique which might not be the case in all datasets
        idx = np.where((combined_points == point).all(axis=1))[0]
        if idx.size > 0:
            indices.append(idx[0])
    return np.array(indices)
