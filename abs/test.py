import h5py
import os
from abs.shape import *


name = 'Cone'
sample_name = f'{name}.hdf5'
file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_hdf5', sample_name)
file_path = os.path.normpath(file_path)


f = h5py.File(file_path, "r")
geo = f['geometry/parts']
group = f['topology/parts']
g_keys = group.keys()
for key in g_keys:
    topo = group[key]


x = Shape(geo, topo)


print('here')



