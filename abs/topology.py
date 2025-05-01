import numpy as np
import h5py
from typing import Any, Dict, List
from abs.winding_number import winding_number


class Topology:

    @staticmethod
    def _get_topo_data(topo_data: h5py.File, entity: str) -> List[Dict[str, Any]]:

        def recursive_hdf5_to_dict(group: h5py.Group) -> Dict[str, Any]:

            return {
                key: recursive_hdf5_to_dict(item) if isinstance(item, h5py.Group) else item[()]
                for key, item in group.items()
            }

        sorted_data = sorted(
            ((int(k), recursive_hdf5_to_dict(topo_data[entity][k])) for k in topo_data[entity].keys()),
            key=lambda x: x[0]
        )

        return [{**data,  **{"id": id}} for id, data in sorted_data]




class Edge(Topology):
    def __init__(self, edge):
        self.curve3d = edge['3dcurve']
        self.end_vertex = int(edge['end_vertex'])
        self.start_vertex = int(edge['start_vertex'])
        self.half_edges = None
        self.id = edge['id']


    def get_length(self):
        if self.curve3d:
            return self.curve3d.get_length()
        else:
            return 0

    def derivative(self, points, order=1):
        return self.curve3d.derivative(points, order)

    def sample(self, sample_points):
        return self.curve3d.sample(sample_points)



class Face(Topology):
    def __init__(self, face):
        self.loops = np.array(face['loops'])
        self.surface = np.int64(face['surface'])
        self.surface_orientation = face['surface_orientation']
        self.trimming_curves_2d = []
        self.shells = None
        self.exact_domain = np.array(face['exact_domain'])
        self.nr_singularities = face['has_singularities']
        self.nr_singularities = face['nr_singularities']
        self.outer_loop = face['outer_loop']
        self.singularities = face['singularities']

        self.id = face['id']


    def normal(self, points):

        if self.surface.shape_name == 'Other':
            return None

        normal_points = self.surface.normal(points)
        if not self.surface_orientation:
            normal_points = -normal_points
        return normal_points

    def get_area(self):
        if self.surface is not None:
            return self.surface.get_area()
        else:
            return 0

    def derivative(self, points, order=1):
        return self.surface.derivative(points, order)

    def sample(self, sample_points):
        return self.surface.sample(sample_points)

    def find_adjacent_faces(self):
        """
        Find the adjacent faces of a face.
        """
        #TODO: fix this!
        adjacent_faces = set()
        for loop in self.loops:
            for halfedge in loop.halfedges:
                if halfedge.mates:
                    # is this correct?
                    for other_loop in halfedge.mates.loops:
                        for other_face in other_loop.faces:
                            adjacent_faces.add(other_face)
        adjacent_faces -= {self}
        return adjacent_faces


    def filter_outside_points(self, uv_points):
        """
        Filter out points that are outside the trimming curve of a face.
        """
        total_winding_numbers = np.zeros((len(uv_points), 1))
        curves = self.trimming_curves_2d
        for poly in curves:
            total_winding_numbers += winding_number(poly, uv_points)

        res = total_winding_numbers > 0.5
        res = res.reshape(-1)
        return res


class Halfedge(Topology):
    def __init__(self, halfedge):
        self.curve2d = np.int64(halfedge['2dcurve'])
        self.edge = np.int64(halfedge['edge'])
        self.orientation_wrt_edge = halfedge['orientation_wrt_edge']
        if halfedge['mates']:
            self.mates = np.array(halfedge['mates'])
        else:
            self.mates = None
        self.loops = None

        self.id = halfedge['id']


    def get_length(self):
        return self.curve2d.get_length()

    def derivative(self, points, order=1):
        return self.curve2d.derivative(points, order)

    def sample(self, sample_points):
        return self.curve2d.sample(sample_points)




class Loop(Topology):
    def __init__(self, loop):
        self.halfedges = np.array(loop['halfedges'])
        self.faces = None

        self.id = loop['id']

class Shell(Topology):
    def __init__(self, shell):
        self.faces = np.array(shell['faces'], dtype=object)
        self.orientation_wrt_solid = shell['orientation_wrt_solid']
        self.solids = None

        self.id = shell['id']




class Solid(Topology):
    def __init__(self, solid):
        self.shells = np.array(solid['shells'])
        self.id = solid['id']
