# shape_core.py
from geometry.curve import Line, Circle, Ellipse, BSplineCurve
from geometry.surface import Plane, Cylinder, Cone, Sphere, Torus, BSplineSurface
from topology.topology import Topology


class ShapeCore:
    def __init__(self, geometry_data, topology_data):
        # Initialization of Geometry and Topology
        self._curves2d, self._curves3d, self._surfaces, self._bbox = [], [], [], []
        self.__init_geometry(geometry_data)
        assert self._curves2d is not None and self._curves3d is not None and self._surfaces is not None, "Error in geometry initialization"
        self._topology = self.__init_topology(topology_data)
        assert self._topology is not None, "Error in topology initialization"

    def _decode_type(self, data):
        return data.get('type')[()].decode('utf-8')

    def _create_curve(self, curve_data):
        curve_type = self._decode_type(curve_data)
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

    def _create_surface(self, surface_data):
        surface_type = self._decode_type(surface_data)
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

    def __init_geometry(self, data):

        for part in data.values():
            for curve_data in part.get('2dcurves', {}).values():
                try:
                    curve = self._create_curve(curve_data)
                    self._curves2d.append(curve)
                except ValueError as e:
                    print(f"Error in 2d curve: {e}")

            for curve_data in part.get('3dcurves', {}).values():
                try: #TODO: Handle other type for curves
                    curve = self._create_curve(curve_data)
                    self._curves3d.append(curve)
                except ValueError as e:
                    print(f"Error in 3d curve: {e}")

            for surface_data in part.get('surfaces', {}).values():
                try:
                    surface = self._create_surface(surface_data)
                    self._surfaces.append(surface)
                except ValueError as e:
                    print(f"Error in surface: {e}")

            self._bbox.append(part.get('bbox'))

    def __init_topology(self, data):
        topology_parts = []

        for data_key in data.keys():
            part = data.get(data_key)
            topo_part = Topology(part)
            topology_parts.append(topo_part)

        return topology_parts
