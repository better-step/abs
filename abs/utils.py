from pathlib import Path
import h5py
import os
from abs.no_name import *


def read_file(file_path):
    assert Path(file_path).exists(), "Please provide valid file path"
    return h5py.File(file_path, 'r')


def get_file(sample_name):
    return os.path.abspath(os.path.join(os.getcwd(), '..', 'abs', 'data', 'sample_hdf5', sample_name))


def save_obj(filename, pts):
    with open(filename, "w") as f:
        if pts.shape[1] == 2:
            [f.write(f"v {pts[i, 0]} {pts[i, 1]} 0\n") for i in range(pts.shape[0])]
        else:
            [f.write(f"v {pts[i, 0]} {pts[i, 1]} {pts[i, 2]}\n") for i in range(pts.shape[0])]


def save_obj_mesh(filename, pts, faces):
    with open(filename, "w") as f:
        if pts.shape[1] == 2:
            [f.write(f"v {pts[i, 0]} {pts[i, 1]} 0\n") for i in range(pts.shape[0])]
        else:
            [f.write(f"v {pts[i, 0]} {pts[i, 1]} {pts[i, 2]}\n") for i in range(pts.shape[0])]

        [f.write(f"f {faces[i, 0]+1} {faces[i, 1]+1} {faces[i, 2]+1}\n") for i in range(faces.shape[0])]


def save_ply(filename, pts, normals=None):
    pts = np.asarray(pts)
    if normals is not None:
        normals = np.asarray(normals)

        if pts.shape[0] != normals.shape[0]:
            raise ValueError("The number of points and normals must be the same")

        if pts.shape[1] != 3 or normals.shape[1] != 3:
            raise ValueError("Both pts and normals must have shape (n, 3)")

        data = np.hstack((pts, normals))

        header = f"""ply
format ascii 1.0
element vertex {data.shape[0]}
property float x
property float y
property float z
property float nx
property float ny
property float nz
end_header
"""
    else:
        if pts.shape[1] != 3:
            raise ValueError("Points must have shape (n, 3)")

        data = pts

        header = f"""ply
format ascii 1.0
element vertex {data.shape[0]}
property float x
property float y
property float z
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        if normals is not None:
            np.savetxt(f, data, fmt='%f %f %f %f %f %f')
        else:
            np.savetxt(f, data, fmt='%f %f %f')
