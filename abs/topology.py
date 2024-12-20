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

        return [data for _, data in sorted_data]




class Edge(Topology):
    def __init__(self, edge):
        if isinstance(edge, dict):
            self.curve3d = edge['3dcurve']
            self.end_vertex = int(edge['end_vertex'])
            self.start_vertex = int(edge['start_vertex'])
            # TODO: don't use the list
            self.half_edges = None

        else:
            self.curve3d = edge.get('3dcurve')
            self.end_vertex = edge.get('end_vertex')
            self.start_vertex = edge.get('start_vertex')
            self.half_edges = None

    def get_length(self):
        return self.curve3d.get_length()

    def derivative(self, points, order=1):
        return self.curve3d.derivative(points, order)

    def sample(self, sample_points):
        return self.curve3d.sample(sample_points)



class Face(Topology):
    def __init__(self, face):
        if isinstance(face, dict):
            self.loops = np.array(face['loops'])
            self.surface = np.int64(face['surface'])
            self.surface_orientation = face['surface_orientation']
            self.trimming_curves_2d = []
            self.shells = None

        else:
            self.loops = face.get('loops')
            self.surface = face.get('surface')
            self.surface_orientation = face.get('surface_orientation')
            self.trimming_curves_2d = []
            self.shells = None

    def normal(self, points):

        if self.surface.shape_name == 'Other':
            return None

        normal_points = self.surface.normal(points)
        if not self.surface_orientation:
            normal_points = -normal_points
        return normal_points

    def get_area(self):
        return self.surface.get_area()

    def derivative(self, points, order=1):
        return self.surface.derivative(points, order)

    def sample(self, sample_points):
        return self.surface.sample(sample_points)

    def find_adjacent_faces(self):
        """
        Find the adjacent faces of a face.
        """
        adjacent_faces = set()
        for loop in self.loops:
            for halfedge in loop.halfedges:
                if halfedge.mates:
                    for mate in halfedge.mates:
                        adjacent_faces.add(mate.face)
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
        if isinstance(halfedge, dict):
            self.curve2d = np.int64(halfedge['2dcurve'])
            self.edge = np.int64(halfedge['edge'])
            self.orientation_wrt_edge = halfedge['orientation_wrt_edge']
            if halfedge['mates']:
                self.mates = np.array(halfedge['mates'])
            else:
                self.mates = None
            self.loops = None

        else:
            self.curve2d = halfedge.get('2dcurve')
            self.edge = halfedge.get('edge')
            self.orientation_wrt_edge = halfedge.get('orientation_wrt_edge')
            if halfedge.get('mates'):
                self.mates = halfedge.get('mates')
            else:
                self.mates = None
            self.loops = None

    def get_length(self):
        return self.curve2d.get_length()

    def derivative(self, points, order=1):
        return self.curve2d.derivative(points, order)

    def sample(self, sample_points):
        return self.curve2d.sample(sample_points)




class Loop(Topology):
    def __init__(self, loop):
        if isinstance(loop, dict):
            self.halfedges = np.array(loop['halfedges'])
            self.faces = None

        else:
            self.halfedges = loop.get('halfedges')
            self.orientation = loop.get('orientation')
            self.faces = None


class Shell(Topology):
    def __init__(self, shell):
        if isinstance(shell, dict):
            self.faces = np.array(shell['faces'], dtype=object)
            self.orientation_wrt_solid = shell['orientation_wrt_solid']
            self.solids = None

        else:
            self.faces = np.array(shell['faces'], dtype=object)
            self.orientation_wrt_solid = shell.get('orientation_wrt_solid')
            self.solids = None



class Solid(Topology):
    def __init__(self, solid):
        if isinstance(solid, dict):
            self.shells = np.array(solid['shells'])

        else:
            self.shells = solid.get('shells')
