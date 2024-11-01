import h5py
import os
from abs.topology_new import *


# this is for reading the hdf5
def recursive_hdf5_to_dict(group):
    data_dict = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            data_dict[key] = recursive_hdf5_to_dict(item)
        else:
            data_dict[key] = item[()]  # Convert dataset to a numpy array
    return data_dict

def _get_topo_data(topo_data, entity):
    data_dict = [(int(k), recursive_hdf5_to_dict(topo_data[entity][k])) for k in topo_data[entity].keys()]
    data_dict.sort(key=lambda x: x[0])
    return [v[1] for v in data_dict]


def _get_edges(edge_data):
    return Edge(edge_data)

def _get_faces(face_data):
    return Face(face_data)




class Shape_test:
    def __init__(self, topology_data):
        # later add the geometry data
        self.Topology = self.Topology(topology_data)

    class Topology:
        def __init__(self, topology_data):
            self._edges = []
            self.__init_topology(topology_data)

        def __init_topology(self, data):

            entity = 'edges'
            edge_data = _get_topo_data(data, entity)
            for edge in edge_data:
                edge = _get_edges(edge)
                self._edges.append(edge)

            entity = 'faces'
            face_data = _get_topo_data(data, entity)
            for face in face_data:
                face = _get_faces(face)
                self._faces.append(face)







# main goes here
name = 'Cone'
sample_name = f'{name}.hdf5'
file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_hdf5', sample_name)
file_path = os.path.normpath(file_path)

with h5py.File(file_path, 'r') as hdf:
    topo = hdf['topology/parts']
    a = list(topo.values())

f = h5py.File(file_path, "r")
group = f['topology/parts']
g_keys = group.keys()
for key in g_keys:
    hdf5_file = group[key]

# entity = 'edges'
# data_dict = [(int(k), recursive_hdf5_to_dict(a[entity][k])) for k in a[entity].keys()]
# data_dict.sort(key=lambda x: x[0])
# result = [v[1] for v in data_dict]

Shape_test(hdf5_file)

print('end')

# f.close()

