from pathlib import Path
import h5py
import os
import numpy as np
import meshio as mio


def read_file(file_path):
    assert Path(file_path).exists(), "Please provide valid file path"
    return h5py.File(file_path, 'r')


def get_file(sample_name):
    return os.path.abspath(os.path.join(Path(__file__), '..', '..', 'data', 'sample_hdf5', sample_name))


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


def save_parts(name, P, S):
    num_parts = len(P)
    os.makedirs('sample_results', exist_ok=True)

    for i in range(num_parts):
        pts = P[i]
        ss = S[i]

        obj_filename = f'sample_results/{name}_part{i}.obj'
        ply_filename = f'sample_results/{name}_part{i}.ply'

        save_obj(obj_filename, pts)
        save_ply(ply_filename, pts, ss)

        print(f'Saved part {i} to {obj_filename} and {ply_filename}')


def save_mesh(file_path, name):
    with h5py.File(file_path, 'r') as hdf:
        vv = []
        ff = []
        n = 0

        for m in hdf['mesh']:
            v = np.array(hdf[f'mesh/{m}']['points'])
            f = np.array(hdf[f'mesh/{m}']['triangle'])

            vv.append(v)
            ff.append(f + n)
            n += len(v)

        if len(vv) > 0:
            v = np.concatenate(vv)
            f = np.concatenate(ff)

            save_obj_mesh(f'sample_results/{name}_mesh.obj', v, f)


def save_to_xyz(points, filename):
    """
    Save 3D points to an .xyz file.

    Args:
    points (array-like): List or NumPy array of 3D points (x, y, z).
    filename (str): The path to the output .xyz file.
    """
    with open(filename, 'w') as f:
        for point in points[0]:
            # Write each point as X Y Z in a new line
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def save_vtu(save_file_path , P):
    m = mio.Mesh(P, cells={"triangle":np.array([np.arange(P.shape[0]), np.arange(P.shape[0]), np.arange(P.shape[0])]).T})
    m.write(save_file_path)
