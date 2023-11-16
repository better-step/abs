import numpy as np
import h5py
from pathlib import Path

def read_file(file_path):
    assert Path(file_path).exists(), "Please provide valid file path"
    return h5py.File(file_path, 'r')

def write_file(file_path, data):
    # Placeholder for file writing function
    pass


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

def save_combined_shapes(shapes, file_name):
    # Filter out non-numpy array items and empty numpy arrays
    valid_shapes = [shape for shape in shapes.values() if shape.size > 0]

    # Check if there are valid shapes to combine
    if not valid_shapes:
        print(f"Warning: No valid points to save in {file_name}.")
        return

    # Combine valid shapes
    combined_points = np.vstack(valid_shapes)

    # Save using the save_points function
    save_points(combined_points, file_name)
