import numpy as np
import h5py


class Topology:

    def normals(self):
        pass


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

        else:
            self._2dcurves = halfedge.get('2dcurve')
            self._edge = halfedge.get('edge')
            self._orientation_wrt_edge = halfedge.get('orientation_wrt_edge')






