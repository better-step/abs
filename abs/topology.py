import numpy as np
import h5py
from typing import Any, Dict, List


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

        else:
            self._3dcurve = edge.get('3dcurve')
            self._end_vertex = edge.get('end_vertex')
            self._start_vertex = edge.get('start_vertex')


class Face(Topology):
    def __init__(self, face):
        if isinstance(face, dict):
            self._loops = np.array(face['loops'])
            self._surface = np.int64(face['surface'])
            self._surface_orientation = face['surface_orientation']

        else:
            self._loops = face.get('loops')
            self._surface = face.get('surface')
            self._surface_orientation = face.get('surface_orientation')

    def normals(self):
        return self._normals

    def sample(self, points):
        return self._geometry.sample(points)


class Halfedge(Topology):
    def __init__(self, halfedge):
        if isinstance(halfedge, dict):
            self._2dcurves = np.int64(halfedge['2dcurve'])
            self._edge = np.int64(halfedge['edge'])
            self._orientation_wrt_edge = halfedge['orientation_wrt_edge']
            if halfedge['mates']:
                self._mates = np.array(halfedge['mates'])
            else:
                self._mates = None

        else:
            self._2dcurves = halfedge.get('2dcurve')
            self._edge = halfedge.get('edge')
            self._orientation_wrt_edge = halfedge.get('orientation_wrt_edge')
            if halfedge.get('mates'):
                self._mates = halfedge.get('mates')
            else:
                self._mates = None


class Loop(Topology):
    def __init__(self, loop):
        if isinstance(loop, dict):
            self._halfedges = np.array(loop['halfedges'])

        else:
            self._halfedges = loop.get('halfedges')
            self._orientation = loop.get('orientation')


class Shell(Topology):
    def __init__(self, shell):
        if isinstance(shell, dict):
            self._faces = np.array(shell['faces'])
            self._orientation_wrt_solid = shell['orientation_wrt_solid']

        else:
            self._faces = shell.get('faces')
            self._orientation_wrt_solid = shell.get('orientation_wrt_solid')


class Solid(Topology):
    def __init__(self, solid):
        if isinstance(solid, dict):
            self._shells = np.array(solid['shells'])

        else:
            self._shells = solid.get('shells')
