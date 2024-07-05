from abs.geometry.curve import *
from abs.geometry.surface import *
from abs.topology import *


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
        raise ValueError(f"Unsupported surface type: {surface_type}")


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
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}")


class Shape:
    def __init__(self, geometry_data, topology_data):
        self.Geometry = self.Geometry(geometry_data)
        self.Topology = self.Topology(topology_data)

    class Geometry:
        def __init__(self, geometry_data):
            self._curves2d, self._curves3d, self._surfaces, self._bbox = [], [], [], []
            self.__init_geometry(geometry_data)

        def __init_geometry(self, data):
            for part in data.values():
                for curve_data in part.get('2dcurves', {}).values():
                    curve = _create_curve(curve_data)
                    self._curves2d.append(curve)

                for curve_data in part.get('3dcurves', {}).values():
                    curve = _create_curve(curve_data)
                    self._curves3d.append(curve)

                for surface_data in part.get('surfaces', {}).values():
                    surface = _create_surface(surface_data)
                    self._surfaces.append(surface)

                self._bbox.append(part.get('bbox'))

    class Topology:
        def __init__(self, topology_data):
            self._topology = None
            self.__init_topology(topology_data)

        def __init_topology(self, data):
            topology_parts = []
            for data_key in data.keys():
                part = data.get(data_key)
                topo_part = Topology(part)
                topology_parts.append(topo_part)
            self._topology = topology_parts
