from abs.topology import *
from abs.curve import *
from abs.surface import *


def _create_surface(surface_data):
    surface_type = surface_data.get('type')[()].decode('utf-8')
    surface_map = {
        'Plane': Plane,
        'Cylinder': Cylinder,
        'Cone': Cone,
        'Sphere': Sphere,
        'Torus': Torus,
        'BSpline': BSplineSurface
    }
    surface_class = surface_map.get(surface_type)
    if surface_class:
        return surface_class(surface_data)
    else:
        print(f"This surface type: {surface_type}, is currently not supported")
        return None


def _create_curve(curve_data):
    curve_type = curve_data.get('type')[()].decode('utf-8')

    curve_map = {
        'Line': Line,
        'Circle': Circle,
        'Ellipse': Ellipse,
        'BSpline': BSplineCurve
    }
    curve_class = curve_map.get(curve_type)
    if curve_class:
        return curve_class(curve_data)

def _get_edges(edge_data):
    return Edge(edge_data)

def _get_faces(face_data):
    return Face(face_data)

def _get_halfedges(halfedge_data):
    return Halfedge(halfedge_data)

def _get_loops(loop_data):
    return Loop(loop_data)



class Shape:
    def __init__(self, geometry_data, topology_data, spacing=200):
        self.Geometry = self.Geometry(geometry_data)
        self.Topology = self.Topology(topology_data)

    class Geometry:
        def __init__(self, geometry_data):
            self._curves2d, self._curves3d, self._surfaces, self._bbox = [], [], [], []
            self.__init_geometry(geometry_data)

        def __init_geometry(self, data):
            for curve_data in data.get('2dcurves', {}).values():
                curve = _create_curve(curve_data)
                self._curves2d.append(curve)

            for curve_data in data.get('3dcurves', {}).values():
                curve = _create_curve(curve_data)
                self._curves3d.append(curve)

            for surface_data in data.get('surfaces', {}).values():
                surface = _create_surface(surface_data)
                self._surfaces.append(surface)

            self._bbox.append(np.array(data.get('bbox')[:]))


    class Topology:
        def __init__(self, topology_data):
            self._edges, self._faces, self._halfedges, self._loops, self._shells, self._solids = [], [], [], [], [], []
            self.__init_topology(topology_data)

        def __init_topology(self, data):

            entity_map = {
                'edges': (self._edges, _get_edges),
                'faces': (self._faces, _get_faces),
                'halfedges': (self._halfedges, _get_halfedges),
                'loops': (self._loops, _get_loops),
                'shells': (self._shells, Shell),
                'solids': (self._solids, Solid)
            }

            for entity, (attr_list, constructor) in entity_map.items():
                entity_data = Topology._get_topo_data(data, entity)
                attr_list.extend(constructor(item) for item in entity_data)

