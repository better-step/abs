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

import numpy as np

def save_ply_color(filename, pts, color=None):
    pts = np.asarray(pts)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Points must have shape (n, 3)")

    has_color = color is not None

    if has_color:
        color = np.asarray(color)
        if color.shape != (pts.shape[0], 3):
            raise ValueError("Color must have the same number of rows as points and exactly 3 columns (R, G, B)")
        if np.issubdtype(color.dtype, np.floating):
            if np.any((color < 0) | (color > 1)):
                raise ValueError("Floating-point color values must be in the range [0, 1]")
            color = (color * 255).astype(np.uint8)  # Convert float [0,1] to uint8 [0,255]
        elif np.issubdtype(color.dtype, np.integer):
            if np.any((color < 0) | (color > 255)):
                raise ValueError("Integer color values must be in the range [0, 255]")
            color = color.astype(np.uint8)
        else:
            raise TypeError("Color values must be either integers (0-255) or floats (0-1)")

        data = np.hstack((pts, color))
        header = f"""ply
format ascii 1.0
element vertex {pts.shape[0]}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        fmt = '%g %g %g %d %d %d'  # XYZ as float, RGB as integers
    else:
        data = pts
        header = f"""ply
format ascii 1.0
element vertex {pts.shape[0]}
property float x
property float y
property float z
end_header
"""
        fmt = '%g %g %g'  # XYZ only

    # Writing to file
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, data, fmt=fmt)




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
        for point in points:
            # Write each point as X Y Z in a new line
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def save_vtu(save_file_path , P):
    m = mio.Mesh(P, cells={"triangle":np.array([np.arange(P.shape[0]), np.arange(P.shape[0]), np.arange(P.shape[0])]).T})
    m.write(save_file_path)


def get_mesh(mesh):

    global_vertices = []
    global_faces = []
    vertex_offset = 0

    for key in mesh:

        sub_mesh = mesh[key]

        vertices = sub_mesh["points"][:]
        if len(vertices) == 0:
            continue
        global_vertices.append(vertices)

        faces = sub_mesh["triangle"][:] + vertex_offset
        global_faces.append(faces)

        vertex_offset += vertices.shape[0]

    global_vertices = np.vstack(global_vertices)
    global_faces = np.vstack(global_faces)

    return global_vertices, global_faces
