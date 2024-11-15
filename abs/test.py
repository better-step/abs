import h5py
import os
from abs.shape import *
from abs.part_processor import *
from abs.utils import *

def get_normal_func(part, topo, points):

    if isinstance(topo, Face):
        return topo.normal(points)
    else:
        return None



name = 'Sphere'
save_type = 'ply'
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

num_samples = 2000
P, S = get_parts(parts, num_samples, get_normal_func)

save_file_path = os.path.join(os.path.dirname(__file__), '..', 'test', 'sample_results', f'{name}.{save_type}')

save_ply(save_file_path, P[0], S[0])
# save_obj(save_file_path, P[0])
#save_vtu(save_file_path, P[0])

print('Finished')



