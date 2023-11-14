# shape_core.py
from geometry.curve import Line, Circle, Ellipse, BSplineCurve
from geometry.surface import Plane, Cylinder, Cone, Sphere, Torus, BSplineSurface
from topology.topology import Topology

class ShapeCore:
    def __init__(self, geometry_data, topology_data):
        # Initialization of Geometry and Topology
        self._curves2d, self._curves3d, self._surfaces, self._bbox= self.__init_geometry(geometry_data)
        self._topology = self.__init_topology(topology_data)

    def __init_geometry(self, data):
        curves2d, curves3d, surfaces, bbox = [], [], [], []

        for part in data:
            for curve_data in part.get('2dcurves', []):
                curves2d.append(self.__create_curve(curve_data))

            for curve_data in part.get('3dcurves', []):
                curves3d.append(self.__create_curve(curve_data))

            for surface_data in part.get('surfaces', []):
                surfaces.append(self.__create_surface(surface_data))

            bbox.append(part.get('bbox'))

        return curves2d, curves3d, surfaces, bbox

    def __create_curve(self, curve_data):
        curve_type = curve_data['type']
        if curve_type == 'Line':
            return Line(curve_data)
        elif curve_type == 'Circle':
            return Circle(curve_data)
        elif curve_type == 'Ellipse':
            return Ellipse(curve_data)
        elif curve_type == 'BSpline':
            return BSplineCurve(curve_data)
        else:
            raise ValueError(f"Unsupported curve type: {curve_type}")

    def __create_surface(self, surface_data):
        surface_type = surface_data['type']
        if surface_type == 'Plane':
            return Plane(surface_data)
        elif surface_type == 'Cylinder':
            return Cylinder(surface_data)
        elif surface_type == 'Cone':
            return Cone(surface_data)
        elif surface_type == 'Sphere':
            return Sphere(surface_data)
        elif surface_type == 'Torus':
            return Torus(surface_data)
        elif surface_type == 'BSpline':
            return BSplineSurface(surface_data)
        else:
            raise ValueError(f"Unsupported surface type: {surface_type}")

    def __init_topology(self, data):
        topology_parts = []

        for part in data:
            topo_part = Topology(part)
            topology_parts.append(topo_part)

        return topology_parts
