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
    if points.shape[1] == 2:
        tmp = np.zeros((points.shape[0], 3))
        tmp[:, 0:2] = points
    else:
        tmp = points

    with open(file_name, "w") as f:
        for i in range(tmp.shape[0]):
            f.write("v {} {} {}\n".format(tmp[i, 0], tmp[i, 1], tmp[i, 2]))
