from pathlib import Path
import os
import meshio as mio
from abs.shape import *
import numpy as np
import h5py


def read_file(file_path):
    assert Path(file_path).exists(), "Please provide valid file path"
    return h5py.File(file_path, 'r')


def get_file(sample_name):
    return os.path.abspath(os.path.join(Path(__file__), '..', '..', 'data', 'sample_hdf5', sample_name))

def read_parts(file_path):
    f = h5py.File(file_path, 'r')
    part = f['parts'].values()

    parts = []
    for i, p in enumerate(part):
        s = Shape(p['geometry'], p['topology'])
        parts.append(s)

    return parts

def read_meshes(file_path):
    f = h5py.File(file_path, 'r')
    part = f['parts'].values()

    meshes = []
    for i, p in enumerate(part):
        s = Shape(p['geometry'], p['topology'])
        mesh_group = p['mesh']
        current_mesh = [None] * len(s.Solid.faces)
        for key in mesh_group:
            submesh = mesh_group[key]
            vertices = submesh['points']
            faces = submesh['triangle']

            current_mesh[int(key)] = {
                'points': vertices,
                'triangle': faces
            }
        meshes.append(current_mesh)

    return meshes

def get_shape(file_path):

    f = h5py.File(file_path, 'r')
    part = f['parts'].values()

    parts = []
    for p in part:

        s = Shape(p['geometry'], p['topology'])
        parts.append(s)

    return parts


def save_obj(filename, pts):
    '''
    Save a set of 3D points to an .obj file.
    '''
    with open(filename, "w") as f:
        if pts.shape[1] == 2:
            [f.write(f"v {pts[i, 0]} {pts[i, 1]} 0\n") for i in range(pts.shape[0])]
        else:
            [f.write(f"v {pts[i, 0]} {pts[i, 1]} {pts[i, 2]}\n") for i in range(pts.shape[0])]


def save_obj_mesh(filename, pts, faces):
    '''
    Save a set of 3D points and faces to an .obj file.
    '''
    with open(filename, "w") as f:
        if pts.shape[1] == 2:
            [f.write(f"v {pts[i, 0]} {pts[i, 1]} 0\n") for i in range(pts.shape[0])]
        else:
            [f.write(f"v {pts[i, 0]} {pts[i, 1]} {pts[i, 2]}\n") for i in range(pts.shape[0])]

        [f.write(f"f {faces[i, 0]+1} {faces[i, 1]+1} {faces[i, 2]+1}\n") for i in range(faces.shape[0])]


def save_ply(filename, P, normals=None):
    '''
    Save a set of 3D points to a .ply file. Optionally, also save normals.
    '''
    total_points = []
    total_normals = []

    # for idx, part in enumerate(P):
    #     if normals:
    #         normal = normals[idx]
    #     for i, pts in enumerate(part):
    #         if normals:
    #             if pts.shape[0] != normal.shape[0]:
    #                 raise ValueError("The number of points and normals must be the same")
    #             if pts.shape[1] != 3 or normal.shape[1] != 3:
    #                 raise ValueError("Both pts and normals must have shape (n, 3)")
    #             total_points.append(pts)
    #             total_normals.append(normal)
    #         else:
    #             if pts.shape[1] != 3:
    #                 raise ValueError("Points must have shape (n, 3)")
    #             total_points.append(pts)


    for i, pts in enumerate(P):
        if normals:
            normal = normals[i]
            if pts.shape[0] != normal.shape[0]:
                raise ValueError("The number of points and normals must be the same")
            if pts.shape[1] != 3 or normal.shape[1] != 3:
                raise ValueError("Both pts and normals must have shape (n, 3)")
            total_points.append(pts)
            total_normals.append(normal)
        else:
            if pts.shape[1] != 3:
                raise ValueError("Points must have shape (n, 3)")
            total_points.append(pts)

    new_pts = np.asarray(total_points)
    if new_pts.shape[2] == 1:
        new_pts = np.squeeze(new_pts, axis=0)
    else:
        new_pts = np.vstack(new_pts)

    if total_normals:
        new_normal = np.asarray(total_normals)
        if new_normal.shape[2] == 1:
            new_normal = np.squeeze(new_normal, axis=0)
        else:
            new_normal = np.vstack(new_normal)

        if new_pts.shape[0] != new_normal.shape[0]:
            raise ValueError("The number of points and normals must be the same")

        if new_pts.shape[1] != 3 or new_normal.shape[1] != 3:
            raise ValueError("Both pts and normals must have shape (n, 3)")

        data = np.hstack((new_pts, new_normal))

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
        if new_pts.shape[1] != 3:
            raise ValueError("Points must have shape (n, 3)")

        data = new_pts

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

def save_to_xyz(points, filename):
    """
    Save 3D points to an .xyz file.
    """
    with open(filename, 'w') as f:
        for point in points:
            # Write each point as X Y Z in a new line
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def save_vtu(save_file_path , P):
    m = mio.Mesh(P, cells={"triangle":np.array([np.arange(P.shape[0]), np.arange(P.shape[0]), np.arange(P.shape[0])]).T})
    m.write(save_file_path)


def get_mesh(meshes):
    global_vertices = []
    global_faces = []
    vertex_offset = 0

    for mesh in meshes:
        for key in range(len(mesh)):

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


def get_mesh_part(mesh):
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


