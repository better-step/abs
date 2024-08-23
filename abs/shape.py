from abs.geometry.curve import *
from abs.geometry.surface import *
from abs.topology import *
from abs.sampling import curve_sampler
from abs.sampling import surface_sampler
from abs.winding_number import winding_number, find_surface_uv_for_curve

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
    # else:
    #     print(f"This curve type: {curve_type}, is currently not supported")
    #     return None


class Shape:
    def __init__(self, geometry_data, topology_data, spacing=1):
        self.Geometry = self.Geometry(geometry_data)
        self.Topology = self.Topology(topology_data)

        self._create_2d_trimming_curves(self.Geometry._curves2d, self.Geometry._curves3d, spacing)

    def filter_outside_points(self, face_index, uv_points):
        """
        Filter out points that are outside the trimming curve of a face.
        """
        total_winding_numbers = np.zeros((len(uv_points), 1))
        curves = self._2d_trimming_curves[face_index]
        for poly in curves:
            # period_u, period_v = self._determine_surface_periodicity(surface)
            period_u = None
            period_v = None

            total_winding_numbers += winding_number(poly, uv_points, period_u=period_u, period_v=period_v)

        res = total_winding_numbers > 0.5
        res = res.reshape(-1)
        return res

    def _create_2d_trimming_curves(self, curves2d, curves3d, spacing):
        """
        Create 2D trimming curves.
        """
        self._2d_trimming_curves = []

        for _, part in enumerate(self.Topology._topology):
            if len(part.solids) == 0:
                for shell_index, _ in enumerate(part.shells):
                    self._process_2d_trimming_curves_for_shell(part, shell_index, curves2d, curves3d, spacing)
            else:
                for _, solid in enumerate(part.solids):
                    for shell_index in solid['shells']:
                        self._process_2d_trimming_curves_for_shell(part, shell_index, curves2d, curves3d, spacing)

    def _process_2d_trimming_curves_for_shell(self, part, shell_index, curves2d, curves3d, spacing):
        shell = part.shells[shell_index]

        for (face_index, _) in shell['faces']:
            face = part.faces[face_index]
            self._2d_trimming_curves += (face_index-len(self._2d_trimming_curves)+1)* [None]
            self._2d_trimming_curves[face_index] = []

            surface_index = face['surface']
            surface = self.Geometry._surfaces[surface_index]

            for loop_id in face['loops']:
                loop = part.loops[loop_id]
                for halfedge_index in loop['halfedges']:
                    halfedge = part.halfedges[halfedge_index]
                    modified_orientation = part.determine_curve_orientation(face_index, halfedge_index)

                    if '2dcurve' in halfedge:
                        curve2d_index = halfedge['2dcurve']
                        curve2d = curves2d[curve2d_index]
                        _, closest_surface_uv_values_of_curve = curve_sampler.uniform_sample(curve2d, spacing, 4, 300)
                        if not modified_orientation:
                            closest_surface_uv_values_of_curve = closest_surface_uv_values_of_curve[::-1]
                    else:
                        surface_uv_values, surface_points = surface_sampler.uniform_sample(surface, spacing)

                        curve3d_index = halfedge['edge']
                        curve3d = curves3d[curve3d_index]


                        # Sample the curve points to get UV values
                        _, curve_points = curve_sampler.uniform_sample(curve3d, spacing)

                        if not modified_orientation:
                            curve_points = curve_points[::-1]
                        # Calculate the nearest UV values on the surface for the curve points
                        closest_surface_uv_values_of_curve = find_surface_uv_for_curve(surface_points, surface_uv_values, curve_points)

                    self._2d_trimming_curves[face_index].append(closest_surface_uv_values_of_curve)


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

            self._bbox.append(data.get('bbox'))

    class Topology:
        def __init__(self, topology_data):
            self._topology = None
            self.__init_topology(topology_data)

        def __init_topology(self, data):
            topology_parts = []
            topo_part = Topology(data)
            topology_parts.append(topo_part)
            self._topology = topology_parts
