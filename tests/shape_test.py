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
    return geo.normal(points)


sample_name = 'cylinder_Hole.hdf5'
file_path = get_file(sample_name)
with h5py.File(file_path, 'r') as hdf:
    geo = hdf['geometry/parts']
    topo = hdf['topology/parts']
    s = Shape(geo, topo)

get_data(s, 10, l_function)
