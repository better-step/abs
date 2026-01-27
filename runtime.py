from abs import read_parts
import time
import h5py
import shutil
from abs.part_processor import sample_parts
from abs.topology import Face
from abs.utils import save_ply
import numpy as np


def _as_flat_list(x, expected_len=None):
    arr = np.array(x)
    flat = arr.reshape(-1).tolist()
    if expected_len is not None and len(flat) != expected_len:
        raise ValueError(f"Expected length {expected_len}, got {len(flat)}")
    return flat

def process_meshes(part):
    mesh_group = part["mesh"]
    ms = list(mesh_group.values())
    meshes = [None] * len(ms)
    for m in ms:
        idx = int(m.name.split("/")[-1])
        pts = np.asarray(m["points"][()], dtype=np.float32)
        tri = np.asarray(m["triangle"][()], dtype=np.int32)
        meshes[idx] = (pts, tri)

    del part["mesh"]
    mesh_out = part.create_group("mesh")

    points_flat = []
    triangles_flat = []
    point_index = [0]
    triangle_index = [0]

    p_start = 0
    t_start = 0
    for pts, tri in meshes:
        pts = np.asarray(pts, dtype=np.float32)
        tri = np.asarray(tri, dtype=np.int32)

        # Flatten (store element offsets)
        points_flat.extend(pts.reshape(-1).tolist())
        triangles_flat.extend(tri.reshape(-1).tolist())

        p_start += pts.size
        t_start += tri.size

        point_index.append(p_start)
        triangle_index.append(t_start)

    mesh_out.create_dataset("points", data=np.asarray(points_flat, dtype=np.float32))
    mesh_out.create_dataset("point_index", data=np.asarray(point_index, dtype=np.int64))
    mesh_out.create_dataset("triangles", data=np.asarray(triangles_flat, dtype=np.int32))
    mesh_out.create_dataset("triangle_index", data=np.asarray(triangle_index, dtype=np.int64))


def process_edges(part):
    es = part['topology']['edges'].values()
    edges = [None]*len(es)
    for edge in es:
        index = int(edge.name.split("/")[-1])
        edges[index] = (int(edge["3dcurve"][()]), int(edge["start_vertex"][()]), int(edge["end_vertex"][()]))

    del part['topology']['edges']
    start = 0
    compressed = []
    edge_index = []
    for e in edges:
        edge_index.append(start)
        compressed.extend(e)
        start += len(e)
    edge_index.append(start)

    part['topology'].create_dataset('edges', data=compressed)
    part['topology'].create_dataset('edge_index', data=edge_index)

def process_halfedges(part):
    hes = part['topology']['halfedges'].values()
    half_edges = [None]*len(hes)
    for he in hes:
        index = int(he.name.split("/")[-1])
        if not isinstance(he["mates"], h5py.Group):
            mates = he["mates"][()].tolist()
        else:
            mates = []
        half_edges[index] = [int(he["2dcurve"][()]), int(he["edge"][()]), int(he["orientation_wrt_edge"][()])] + mates

    del part['topology']['halfedges']
    start = 0
    compressed = []
    half_edge_index = []
    for he in half_edges:
        half_edge_index.append(start)
        compressed.extend(he)
        start += len(he)
    half_edge_index.append(start)  # add end index

    part['topology'].create_dataset('halfedges', data=compressed)
    part['topology'].create_dataset('halfedge_index', data=half_edge_index)

def process_loops(part):
    lps = part['topology']['loops'].values()
    loops = [None]*len(lps)
    for lp in lps:
        index = int(lp.name.split("/")[-1])
        loops[index] = lp["halfedges"][()].tolist()

    del part['topology']['loops']
    start = 0
    compressed = []
    loop_index = []
    for lp in loops:
        loop_index.append(start)
        compressed.extend(lp)
        start += len(lp)
    loop_index.append(start)  # add end index

    part['topology'].create_dataset('loops', data=compressed)
    part['topology'].create_dataset('loop_index', data=loop_index)

def process_shells(part):
    shs = part['topology']['shells'].values()
    shells = [None]*len(shs)
    for sh in shs:
        index = int(sh.name.split("/")[-1])
        shells[index] = [int(sh["orientation_wrt_solid"][()])] + [int(f) for fs in sh["faces"][()].tolist() for f in fs]

    del part['topology']['shells']
    start = 0
    compressed = []
    shell_index = []
    for sh in shells:
        shell_index.append(start)
        compressed.extend(sh)
        start += len(sh)
    shell_index.append(start)

    part['topology'].create_dataset('shells', data=compressed)
    part['topology'].create_dataset('shell_index', data=shell_index)

def process_solids(part):
    sls = part['topology']['solids'].values()
    solids = [None]*len(sls)
    for sl in sls:
        index = int(sl.name.split("/")[-1])
        solids[index] = sl["shells"][()].tolist()

    del part['topology']['solids']
    start = 0
    compressed = []
    solid_index = []
    for sl in solids:
        solid_index.append(start)
        compressed.extend(sl)
        start += len(sl)
    solid_index.append(start)

    part['topology'].create_dataset('solids', data=compressed)
    part['topology'].create_dataset('solid_index', data=solid_index)

def process_faces(part):
    fs = part['topology']['faces'].values()
    faces = [None]*len(fs)
    for f in fs:
        index = int(f.name.split("/")[-1])
        ed = f["exact_domain"][()].tolist()
        hs = int(f["has_singularities"][()])
        lps = f["loops"][()].tolist()
        ns = int(f["nr_singularities"][()])
        os = int(f["outer_loop"][()])

        sings_grp = f["singularities"]
        items = []
        for sg_name in sings_grp.keys():
            try:
                items.append((int(sg_name), sings_grp[sg_name]))
            except ValueError:
                items.append((sg_name, sings_grp[sg_name]))
        items.sort(key=lambda t: t[0])

        flat = []
        for _, sg in items:
            flat.extend(_as_flat_list(sg["first2d"][()], expected_len=2))
            flat.append(float(sg["firstpar"][()]))

            flat.extend(_as_flat_list(sg["last2d"][()], expected_len=2))
            flat.append(float(sg["lastpar"][()]))

            flat.extend(_as_flat_list(sg["point2d"][()], expected_len=2))
            flat.extend(_as_flat_list(sg["point3d"][()], expected_len=3))

            flat.append(float(sg["precision"][()]))
            flat.append(float(sg["rank"][()]))
            flat.append(int(bool(sg["uiso"][()])))

        srf = int(f["surface"][()])
        so = int(f["surface_orientation"][()])
        faces[index] = ed + [hs, ns, os, srf, so]+ flat + lps

    del part['topology']['faces']
    start = 0
    compressed = []
    face_index = []
    for f in faces:
        face_index.append(start)
        compressed.extend(f)
        start += len(f)
    face_index.append(start)  # add end index

    part['topology'].create_dataset('faces', data=compressed)
    part['topology'].create_dataset('face_index', data=face_index)


def process_curves(part, curve_type, has_trafo):
    cs = part['geometry'][curve_type].values()
    curves = [None]*len(cs)
    for c in cs:
        index = int(c.name.split("/")[-1])
        tt = c["type"][()].decode('utf8')
        if tt == "Line":  # line
            tt = 0
            curves[index] = [tt] + \
            c["interval"][()].tolist() + \
            c["location"][()].tolist() + \
            c["direction"][()].tolist()
        elif tt == "Circle":  # circle
            tt = 1
            curves[index] = [tt] + \
            c["interval"][()].tolist() + \
            c["location"][()].tolist() + \
            c["x_axis"][()].tolist() + \
            c["y_axis"][()].tolist()
            if 'z_axis' in c:
                curves[index] += c["z_axis"][()].tolist()
            curves[index] += [c["radius"][()]]
        elif tt == "Ellipse":  # ellipse
            tt = 2
            curves[index] = [tt] + \
            c["interval"][()].tolist() + \
            c["focus1"][()].tolist() + \
            c["focus2"][()].tolist() + \
            c["x_axis"][()].tolist() + \
            c["y_axis"][()].tolist()
            if 'z_axis' in c:
                curves[index] += c["z_axis"][()].tolist()
            curves[index] += [c["maj_radius"][()], c["min_radius"][()]]
        elif tt == "BSpline":  # BSplineCurve
            tt = 3
            curves[index] = [tt]
            pshape = c["poles"][()].shape
            poles = c["poles"][()].flatten().tolist()
            knots = c["knots"][()].flatten().tolist()
            weights = c["weights"][()].flatten().tolist()
            curves[index] += c["interval"][()].tolist() + \
                [c["degree"][()],
                c["continuity"][()],
                c["rational"][()],
                c["periodic"][()],
                c["closed"][()]] + \
                [len(poles)] + list(pshape) + poles + \
                [len(knots)] + knots + \
                [len(weights)] + weights
        elif tt == "Other": # Other
            tt = 4
            curves[index] = [tt]
        else:
            raise ValueError(f"Unknown curve type: {tt}")

        if has_trafo:
            curves[index] += c["transform"][()].flatten().tolist()

    del part['geometry'][curve_type]
    start = 0
    compressed = []
    curve_index = []
    for c in curves:
        curve_index.append(start)
        compressed.extend(c)
        start += len(c)
    curve_index.append(start)  # add end index


    part['geometry'].create_dataset(curve_type, data=compressed)
    part['geometry'].create_dataset(curve_type + '_index', data=curve_index)

def process_surfaces(part):
    ss = part['geometry']['surfaces'].values()
    surfaces = [None]*len(ss)
    for s in ss:
        index = int(s.name.split("/")[-1])
        tt = s["type"][()].decode('utf8')
        td = s["trim_domain"][()].flatten().tolist()
        if tt == "Plane":  # plane
            tt = 0
            surfaces[index] = [tt] + td + \
            s["location"][()].tolist() + \
            s["coefficients"][()].tolist() + \
            s["x_axis"][()].tolist() + \
            s["y_axis"][()].tolist() + \
            s["z_axis"][()].tolist()
        elif tt == "Cylinder":  # cylinder
            tt = 1
            surfaces[index] = [tt] + td + \
            s["location"][()].tolist() + \
            [s["radius"][()]] + \
            s["coefficients"][()].tolist() + \
            s["x_axis"][()].tolist() + \
            s["y_axis"][()].tolist() + \
            s["z_axis"][()].tolist()
        elif tt == "Cone":  # cone
            tt = 2
            surfaces[index] = [tt] + td + \
            s["location"][()].tolist() + \
            [s["radius"][()]] + \
            s["coefficients"][()].tolist() + \
            s["apex"][()].tolist() + \
            [s["angle"][()]] + \
            s["x_axis"][()].tolist() + \
            s["y_axis"][()].tolist() + \
            s["z_axis"][()].tolist()
        elif tt == "Sphere":  # sphere
            tt = 3
            xaxis = np.array(s.get('x_axis')[()]).reshape(-1, 1).T
            yaxis = np.array(s.get('y_axis')[()]).reshape(-1, 1).T
            if 'z_axis' in s:
                zaxis = np.array(s.get('z_axis')[()]).reshape(-1, 1).T
            else:
                zaxis = np.cross(xaxis, yaxis)

            surfaces[index] = [tt] + td + \
            s["location"][()].tolist() + \
            [s["radius"][()]] + \
            s["coefficients"][()].tolist() + \
            xaxis.ravel().tolist() + \
            yaxis.ravel().tolist() + \
            zaxis.ravel().tolist()
        elif tt == "Torus":
            tt = 4
            surfaces[index] = [tt] + td + \
            s["location"][()].tolist() + \
            [s["max_radius"][()], s["min_radius"][()]] + \
            s["x_axis"][()].tolist() + \
            s["y_axis"][()].tolist() + \
            s["z_axis"][()].tolist()
        elif tt == "BSpline":
            tt = 5
            surfaces[index] = [tt] + td
            face_domain = s["face_domain"][()].ravel().tolist()
            pshape = s["poles"][()].shape
            poles = s["poles"][()].ravel().tolist()
            uknots = s["u_knots"][()].ravel().tolist()
            vknots = s["v_knots"][()].ravel().tolist()
            weights_obj = s["weights"]
            if isinstance(weights_obj, h5py.Group):
                weights = np.column_stack([weights_obj[str(i)][()] for i in range(len(weights_obj))])
            else:
                weights = weights_obj[()]

            wshape = weights.shape

            surfaces[index] += [
                s["u_degree"][()],
                s["v_degree"][()],
                s["continuity"][()],
                s["u_rational"][()],
                s["v_rational"][()],
                s["u_periodic"][()],
                s["v_periodic"][()],
                s["u_closed"][()],
                s["v_closed"][()],
                s["is_trimmed"][()],
                len(face_domain)] + \
                face_domain + \
                [len(poles)] + list(pshape) + poles + \
                [len(uknots)] + uknots + \
                [len(vknots)] + vknots + \
                [weights.size] + list(wshape) + weights.ravel().tolist()
        elif tt == "Extrusion":  # Extrusion
            tt = 6
            surfaces[index] = [tt] + td + \
            s["direction"][()].tolist() + [s["curve"][()]]
        elif tt == "Revolution":  # Revolution
            tt = 7
            surfaces[index] = [tt] + td + \
            s["location"][()].tolist() + \
            s["z_axis"][()].tolist() + [s["curve"][()]]
        elif tt == "Offset": # Other
            tt = 8
            surfaces[index] = [tt] + td + [s["value"][()], s["surface"][()]]
        elif tt == "Other": # Other
            tt = 9
            surfaces[index] = [tt] + td
        else:
            raise ValueError(f"Unknown surface type: {tt}")

        surfaces[index] += s["transform"][()].flatten().tolist()

    del part['geometry']['surfaces']

    start = 0
    compressed = []
    surface_index = []
    for s in surfaces:
        surface_index.append(start)
        compressed.extend(s)
        start += len(s)
    surface_index.append(start)  # add end index

    part['geometry'].create_dataset('surfaces', data=compressed)
    part['geometry'].create_dataset('surfaces_index', data=surface_index)


if __name__ == "__main__":
    path = '/home/nafiseh/Documents/abs/data/sample_hdf5/Cone.hdf5'

    create = True

    def compute_labels(part, topo, points ):
        if isinstance(topo, Face): return np.ones((points.shape[0], 1), dtype=np.float32)
        else : return None

    if not create:
        path = path + '_new.hdf5'
        start_time = time.time()
        p = read_parts(path)
        end_time = time.time()
        print(f"Loaded in {end_time - start_time} seconds")

        start_time = time.time()
        P, S = sample_parts(p, 10000, compute_labels)
        end_time = time.time()
        print(f"samples in {end_time - start_time} seconds")
        save_ply('test.ply', P)
    else:
        shutil.copyfile(path, path + '_new.hdf5')
        with h5py.File(path + '_new.hdf5', 'r+') as f:
            f['parts'].attrs["version"] = "3.0"
            for part in f['parts'].values():
                process_meshes(part)
                process_edges(part)
                process_halfedges(part)
                process_loops(part)
                process_shells(part)
                process_solids(part)
                process_faces(part)

                data = ((False, '2dcurves'), (True, '3dcurves'))
                for has_trafo, curve_type in data:
                    process_curves(part, curve_type, has_trafo)

                process_surfaces(part)
