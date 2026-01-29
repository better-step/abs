from pathlib import Path
import meshio as mio
import numpy as np
import h5py
from .shape import Shape


def read_parts(file_path):
    """Read all parts from an HDF5 file and construct Shape objects."""
    f = h5py.File(file_path, 'r')
    part = f['parts'].values()
    version = f['parts'].attrs.get('version')
    parts = []
    for i, p in enumerate(part):
        s = Shape(p['geometry'], p['topology'], version)
        parts.append(s)
    return parts


def read_meshes(file_path):
    """Read pre-computed mesh data (points and triangles) for each face of each part from an HDF5 file."""
    f = h5py.File(file_path, 'r')
    part = f['parts'].values()
    version = f['parts'].attrs.get('version')
    meshes = []
    for i, p in enumerate(part):
        s = Shape(p['geometry'], p['topology'], version)
        if not hasattr(s, 'faces') or not s.faces:
            meshes.append([])
            continue
        if version == '2.0':
            mesh_group = p['mesh']
            current_mesh = [None] * len(s.faces)
            for key in mesh_group:
                submesh = mesh_group[key]
                vertices = submesh['points']
                faces = submesh['triangle']
                current_mesh[int(key)] = {
                    'points': vertices,
                    'triangle': faces
                }
            meshes.append(current_mesh)
        elif version == '3.0':
            mesh_group = p['mesh']
            all_points = mesh_group['points'][()]
            all_tris = mesh_group['triangles'][()]
            p_idx = mesh_group['point_index'][()]
            t_idx = mesh_group['triangle_index'][()]
            n_meshes = len(p_idx) - 1
            current_mesh = [None] * n_meshes
            for mi in range(n_meshes):
                ps = all_points[p_idx[mi]:p_idx[mi + 1]].reshape(-1, 3).astype(np.float32, copy=False)
                ts = all_tris[t_idx[mi]:t_idx[mi + 1]].reshape(-1, 3).astype(np.int32, copy=False)
                current_mesh[mi] = {
                    'points': ps,
                    'triangle': ts
                }
            meshes.append(current_mesh)
    return meshes

# -------------------------
# Saving utilities
# -------------------------

def save_obj_points(filename, pts):
    """Save a set of 2D/3D points to an .obj file."""
    with open(filename, "w") as f:
        if pts.shape[1] == 2:
            for i in range(pts.shape[0]):
                f.write(f"v {pts[i, 0]} {pts[i, 1]} 0\n")
        else:
            for i in range(pts.shape[0]):
                f.write(f"v {pts[i, 0]} {pts[i, 1]} {pts[i, 2]}\n")


def save_obj_mesh(filename, pts, faces):
    """Save a set of 3D points and faces to an .obj file."""
    if pts.shape[0] == 0:
        print("Skipping saving meshes: mesh is empty")
        return
    with open(filename, "w") as f:
        if pts.shape[1] == 2:
            for i in range(pts.shape[0]):
                f.write(f"v {pts[i, 0]} {pts[i, 1]} 0\n")
        else:
            for i in range(pts.shape[0]):
                f.write(f"v {pts[i, 0]} {pts[i, 1]} {pts[i, 2]}\n")
        for i in range(faces.shape[0]):
            f.write(f"f {faces[i, 0] + 1} {faces[i, 1] + 1} {faces[i, 2] + 1}\n")



def save_ply(filename, P, normals=None):
    '''
    Save a set of 3D points to a .ply file. Optionally, also save normals.
    '''
    total_points = []
    total_normals = []

    for i, pts in enumerate(P):
        if (pts.shape[0] == 0):
            continue
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
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def save_vtu(save_file_path , P):
    m = mio.Mesh(P, cells={"triangle":np.array([np.arange(P.shape[0]), np.arange(P.shape[0]), np.arange(P.shape[0])]).T})
    m.write(save_file_path)

def get_mesh(meshes):
    global_vertices = []
    global_faces = []
    vertex_offset = 0
    for mesh in meshes:
        for sub_mesh in mesh:
            if sub_mesh is None:
                continue
            vertices = sub_mesh["points"][:]
            if len(vertices) == 0:
                continue
            global_vertices.append(vertices)
            faces = sub_mesh["triangle"][:] + vertex_offset
            global_faces.append(faces)
            vertex_offset += vertices.shape[0]
    if global_vertices:
        global_vertices = np.vstack(global_vertices)
    else:
        global_vertices = np.empty((0, 3))
    if global_faces:
        global_faces = np.vstack(global_faces)
    else:
        global_faces = np.empty((0, 3), dtype=int)
    return global_vertices, global_faces

def get_mesh_per_part(meshes):

    global_vertices = []
    global_faces = []
    for mesh in meshes:
        V_list = []
        F_list = []
        offset = 0
        for sub_mesh in mesh:
            if sub_mesh is None:
                continue
            vertices = np.asarray(sub_mesh["points"], dtype=np.float32).reshape(-1, 3)
            faces = np.asarray(sub_mesh["triangle"], dtype=np.int32).reshape(-1, 3)
            if vertices.shape[0] == 0 or faces.shape[0] == 0:
                continue
            V_list.append(vertices)
            F_list.append(faces + offset)
            offset += vertices.shape[0]
        if V_list:
            Vp = np.vstack(V_list)
            Fp = np.vstack(F_list) if F_list else np.zeros((0, 3), dtype=np.int32)
        else:
            Vp = np.zeros((0, 3), dtype=np.float32)
            Fp = np.zeros((0, 3), dtype=np.int32)
        global_vertices.append(Vp)
        global_faces.append(Fp)
    return global_vertices, global_faces
