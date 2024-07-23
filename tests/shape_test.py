from pathlib import Path
import h5py
import os
from abs.shape import Shape
from abs.no_name import *


def read_file(file_path):
    assert Path(file_path).exists(), "Please provide valid file path"
    return h5py.File(file_path, 'r')


def get_file(sample_name):
    return os.path.abspath(os.path.join(os.getcwd(), '..', 'abs', 'data', 'sample_hdf5', sample_name))


def l_function(shape, geo, points):
    # get the normals
    if geo._shape_name == 'Circle' and len(geo._interval[0]) == 2:
        return None
    if geo._shape_name == 'Ellipse' and len(geo._interval[0]) == 2:
        return None
    if geo._shape_name == 'BSpline' and len(geo._interval[0]) == 2:
        return None
    else:
        normal_points = geo.normal(points)
        return normal_points


sample_name = 'cylinder_Hole.hdf5'
file_path = get_file(sample_name)
with h5py.File(file_path, 'r') as hdf:
    geo = hdf['geometry/parts']
    topo = hdf['topology/parts']
    s = Shape(geo, topo)

# ss1, pts1 = get_data(s, 10, l_function)
ss, pts =new_get_data(s, 10, l_function)
