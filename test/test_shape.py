from abs.shape import Shape
from abs.utils import *


def l_function(shape, geo, points):
    if geo._shape_name == 'Circle' and len(geo._interval[0]) == 2:
        return None
    if geo._shape_name == 'Ellipse' and len(geo._interval[0]) == 2:
        return None

    if geo._shape_name == 'BSpline':
        try:
            if len(geo._interval[0]) == 2:
                return None
        except AttributeError:
            pass


    normal_points = geo.normal(points)
    return normal_points


def l_function_labels(shape, geo, points):
    labels = geo._shape_name(points)
    return labels

# testing start here
name = "Ellipse"
sample_name = f'{name}.hdf5'
base_name = os.path.splitext(sample_name)[0]

file_path = get_file(sample_name)
with h5py.File(file_path, 'r') as hdf:
    geo = hdf['geometry/parts']
    topo = hdf['topology/parts']
    parts = []
    for i in range(len(geo)):
        s = Shape(list(geo.values())[i], list(topo.values())[i])
        parts.append(s)


pts, ss  = get_data_parts(parts, 1000, l_function)
# pts, ss = get_data(s, 10000, l_function)
# pts, ss = get_data_test(s, 10000, l_function)
#pts, ss = get_data_geo(s, 30000, l_function)


save_obj(f'sample_results/{name}.obj', pts)
save_ply(f'sample_results/{name}.ply', pts, ss)


with h5py.File(file_path, 'r') as hdf:
    vv = []
    ff = []
    n = 0

    for m in hdf['mesh']:
        v = np.array(hdf[f'mesh/{m}']['points'])
        f = np.array(hdf[f'mesh/{m}']['triangle'])

        vv.append(v)
        ff.append(f + n)
        n += len(v)

    if len(vv) > 0:
        v = np.concatenate(vv)
        f = np.concatenate(ff)

        save_obj_mesh(f'sample_results/{name}_mesh.obj', v, f)
