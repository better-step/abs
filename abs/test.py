import h5py
import os
from abs.shape import *
from abs.part_processor import *


def get_normal_func(shape, geo, points):
    try:
        if len(geo._interval[0]) == 2 or geo._shape_name in 'Other':
            return None
    except AttributeError:
        pass


name = 'Ellipse'
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

num_samples = 1000
P, S = get_parts(parts, num_samples, get_normal_func)

print('Finished')



