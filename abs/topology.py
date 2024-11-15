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
            self._3dcurve = edge['3dcurve']
            self._end_vertex = int(edge['end_vertex'])
            self._start_vertex = int(edge['start_vertex'])
            self._half_edges = []

        else:
            self._3dcurve = edge.get('3dcurve')
            self._end_vertex = edge.get('end_vertex')
            self._start_vertex = edge.get('start_vertex')
            self._half_edges = []

    def length(self):
        return self._3dcurve.length()

    def derivative(self, points, order=1):
        return self._3dcurve.derivative(points, order)

    def sample(self, sample_points):
        return self._3dcurve.sample(sample_points)



class Face(Topology):
    def __init__(self, face):
        if isinstance(face, dict):
            self._loops = np.array(face['loops'])
            self._surface = np.int64(face['surface'])
            self._surface_orientation = face['surface_orientation']
            self._2d_trimming_curves = []
            self._shells = []

        else:
            self._loops = face.get('loops')
            self._surface = face.get('surface')
            self._surface_orientation = face.get('surface_orientation')
            self._2d_trimming_curves = []
            self._shells = []

    def normal(self, points):

        if self._surface._shape_name == 'Other':
            return None

        normal_points = self._surface.normal(points)
        if not self._surface_orientation:
            normal_points = -normal_points
        return normal_points

    def area(self):
        return self._surface.area()

    def derivative(self, points, order=1):
        return self._surface.derivative(points, order)

    def sample(self, sample_points):
        return self._surface.sample(sample_points)

    def find_adjacent_faces(self):
        """
        Find the adjacent faces of a face.
        """
        adjacent_faces = set()
        for loop in self._loops:
            for halfedge in loop._halfedges:
                if halfedge._mates:
                    for mate in halfedge._mates:
                        adjacent_faces.add(mate._face)
        return adjacent_faces


    def filter_outside_points(self, uv_points):
        """
        Filter out points that are outside the trimming curve of a face.
        """
        total_winding_numbers = np.zeros((len(uv_points), 1))
        curves = self._2d_trimming_curves
        for poly in curves:
            # period_u, period_v = self._determine_surface_periodicity(surface)
            period_u = None
            period_v = None

            total_winding_numbers += winding_number(poly, uv_points, period_u=period_u, period_v=period_v)

        res = total_winding_numbers > 0.5
        res = res.reshape(-1)
        return res




class Halfedge(Topology):
    def __init__(self, halfedge):
        if isinstance(halfedge, dict):
            self._2dcurve = np.int64(halfedge['2dcurve'])
            self._edge = np.int64(halfedge['edge'])
            self._orientation_wrt_edge = halfedge['orientation_wrt_edge']
            if halfedge['mates']:
                self._mates = np.array(halfedge['mates'])
            else:
                self._mates = None
            self._loops = []

        else:
            self._2dcurve = halfedge.get('2dcurve')
            self._edge = halfedge.get('edge')
            self._orientation_wrt_edge = halfedge.get('orientation_wrt_edge')
            if halfedge.get('mates'):
                self._mates = halfedge.get('mates')
            else:
                self._mates = None
            self._loops = []

    def length(self):
        return self._2dcurve.length()

    def derivative(self, points, order=1):
        return self._2dcurve.derivative(points, order)

    def sample(self, sample_points):
        return self._2dcurve.sample(sample_points)




class Loop(Topology):
    def __init__(self, loop):
        if isinstance(loop, dict):
            self._halfedges = np.array(loop['halfedges'])
            self._faces = []

        else:
            self._halfedges = loop.get('halfedges')
            self._orientation = loop.get('orientation')
            self._faces = []


class Shell(Topology):
    def __init__(self, shell):
        if isinstance(shell, dict):
            self._faces = np.array(shell['faces'], dtype=object)
            self._orientation_wrt_solid = shell['orientation_wrt_solid']
            self._solids = []

        else:
            self._faces = np.array(shell['faces'], dtype=object)
            self._orientation_wrt_solid = shell.get('orientation_wrt_solid')
            self._solids = []



class Solid(Topology):
    def __init__(self, solid):
        if isinstance(solid, dict):
            self._shells = np.array(solid['shells'])

        else:
            self._shells = solid.get('shells')
