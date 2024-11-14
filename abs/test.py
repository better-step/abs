import h5py
import os
import numpy as np
from abs.shape import *
from abs.part_processor import *
from abs.utils import *
import meshio as mio

def get_normal_func(part, topo, points):

    try:
        if hasattr(topo, '_3dcurve'):
            return None
        if topo._surface._shape_name == 'Other':
            return None
    except AttributeError:
        pass

    surf = topo._surface
    surf_orientation = topo._surface_orientation
    shell_orienttion = part.Solid.shells[0]._orientation_wrt_solid
    normal_points = surf.normal(points)

    if not surf_orientation and shell_orienttion:
        normal_points = -normal_points

    return normal_points


name = 'Sphere'
sample_name = f'{name}.hdf5'
file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_hdf5', sample_name)
file_path = os.path.normpath(file_path)


f = h5py.File(file_path, "r")
geo = f['geometry/parts']
group = f['topology/parts']
parts = []
for i in range(len(geo)):
    s = Shape(list(geo.values())[i], list(group.values())[i])
    parts.append(s)

num_samples = 10000
P, S = get_parts(parts, num_samples, get_normal_func)

save_file_path = os.path.join(os.path.dirname(__file__), '..', 'test', 'sample_results', f'{name}.ply')
save_ply(save_file_path, P[0], S[0])
# save_obj(save_file_path, P[0])

#m = mio.Mesh(P[0], cells={"triangle":np.array([np.arange(P[0].shape[0]), np.arange(P[0].shape[0]), np.arange(P[0].shape[0])]).T})
#m.write(save_file_path)

print('Finished')



