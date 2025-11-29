from abs import read_parts
import time
import h5py
import shutil


path = '/Users/teseo/Downloads/abs/assembly 3.hdf5' # Loaded in 167.06935691833496 seconds
# path = '/Users/teseo/data/abc/Hdf5MeshSampler/data/sample_hdf5/Box.hdf5' # Loaded in 0.028881072998046875 seconds
# Loaded in 105.97194314002991 seconds
# path = '/Users/teseo/data/abc/Hdf5MeshSampler/data/sample_hdf5/Cone.hdf5'
# path = '/Users/teseo/data/abc/Hdf5MeshSampler/data/sample_hdf5/Circle.hdf5'
# path = '/Users/teseo/data/abc/Hdf5MeshSampler/data/sample_hdf5/Cylinder_Hole_Fillet_Chamfer.hdf5'

create = False

if not create:
    path = path + '_new.hdf5'
    start_time = time.time()
    p = read_parts(path)
    end_time = time.time()
    print(f"Loaded in {end_time - start_time} seconds")
else:
    shutil.copyfile(path, path + '_new.hdf5')
    with h5py.File(path + '_new.hdf5', 'r+') as f:
        for part in f['parts'].values():
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
        edge_index.append(start)  # add end index

        part['topology'].create_dataset('edges', data=compressed)
        part['topology'].create_dataset('edge_index', data=edge_index)


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
        shell_index.append(start)  # add end index

        part['topology'].create_dataset('shells', data=compressed)
        part['topology'].create_dataset('shell_index', data=shell_index)

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
        solid_index.append(start)  # add end index

        part['topology'].create_dataset('solids', data=compressed)
        part['topology'].create_dataset('solid_index', data=solid_index)

        fs = part['topology']['faces'].values()
        faces = [None]*len(fs)
        for f in fs:
            index = int(f.name.split("/")[-1])
            ed = f["exact_domain"][()].tolist()
            hs = int(f["has_singularities"][()])
            lps = f["loops"][()].tolist()
            ns = int(f["nr_singularities"][()])
            os = int(f["outer_loop"][()])
            # sgs = f["singularities"].values() #TODO
            srf = int(f["surface"][()])
            so = int(f["surface_orientation"][()])
            faces[index] = ed + [hs, ns, os, srf, so] + lps

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




        ########### curves
        data = ((False, '2dcurves'), (True, '3dcurves'))

        for has_trafo, curve_type in data:
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