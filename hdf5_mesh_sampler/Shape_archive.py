# from geometry.curve import Line, Circle, Ellipse, BSplineCurve
# from geometry.surface import Plane, Cylinder, Cone, Sphere, Torus, BSplineSurface
# from topology.topology import Topology
# import numpy as np
# from winding_number import calculate_winding_numbers
#
#
#
#
# class Shape:
#     def __init__(self, geometry_data, topology_data):
#         # Initialization of Geometry and Topology
#         self._curves2d, self._curves3d, self._surfaces = self.__init_geometry(geometry_data)
#         self._topology = self.__init_topology(topology_data)
#
#     def __init_geometry(self, data):
#         curves2d, curves3d, surfaces = [], [], []
#
#         for part in data:
#             for curve_data in part['2dcurves']:
#                 curves2d.append(self.__create_curve(curve_data))
#
#             for curve_data in part['3dcurves']:
#                 curves3d.append(self.__create_curve(curve_data))
#
#             for surface_data in part['surfaces']:
#                 surfaces.append(self.__create_surface(surface_data))
#
#         return curves2d, curves3d, surfaces
#
#     def __create_curve(self, curve_data):
#         curve_type = curve_data['type']
#         if curve_type == 'Line':
#             return Line(curve_data)
#         elif curve_type == 'Circle':
#             return Circle(curve_data)
#         elif curve_type == 'Ellipse':
#             return Ellipse(curve_data)
#         elif curve_type == 'BSpline':
#             return BSplineCurve(curve_data)
#         # Add more curve types if needed
#         raise ValueError(f"Unsupported curve type: {curve_type}")
#
#     def __create_surface(self, surface_data):
#         surface_type = surface_data['type']
#         if surface_type == 'Plane':
#             return Plane(surface_data)
#         elif surface_type == 'Cylinder':
#             return Cylinder(surface_data)
#         elif surface_type == 'Cone':
#             return Cone(surface_data)
#         elif surface_type == 'Sphere':
#             return Sphere(surface_data)
#         elif surface_type == 'Torus':
#             return Torus(surface_data)
#         elif surface_type == 'BSpline':
#             return BSplineSurface(surface_data)
#         # Add more surface types if needed
#         raise ValueError(f"Unsupported surface type: {surface_type}")
#
#     def __init_topology(self, data):
#         topology_parts = []
#
#         for part in data:
#             topo_part = Topology(part)
#             topo_part.process_data()
#             topology_parts.append(topo_part)
#
#         return topology_parts
#
#     def sample_curve(self, curve_index, sampler):
#         """
#         Sample points on a 3D curve.
#
#         Args:
#         curve_index (int): Index of the curve to be sampled.
#         sampler (CurveSampler): An instance of a CurveSampler.
#
#         Returns:
#         numpy.ndarray: Array of sampled points on the curve.
#         """
#         assert 0 <= curve_index < len(self._curves3d), "Invalid curve_index"
#         curve = self._curves3d[curve_index]
#         return sampler.sample(curve)
#
#     def sample_surface(self, surface_index, sampler, winding_number_calculator):
#         """
#         Sample points on a surface, considering trimming curves and winding numbers.
#
#         Args:
#         surface_index (int): Index of the surface to be sampled.
#         sampler (SurfaceSampler): An instance of a SurfaceSampler.
#         winding_number_calculator (function): A function to calculate winding numbers.
#
#         Returns:
#         numpy.ndarray: Array of sampled points on the surface.
#         """
#         assert 0 <= surface_index < len(self._surfaces), "Invalid surface_index"
#         surface = self._surfaces[surface_index]
#
#         # Sample points on the surface
#         sampled_points = sampler.sample(surface)
#
#         # Filter points based on winding number
#         filtered_points = []
#         for point in sampled_points:
#             wn = winding_number_calculator(point, self._trimming_curves[surface_index])
#             if wn > 0.5:  # Assuming the winding number threshold is 0.5
#                 filtered_points.append(point)
#
#         return np.array(filtered_points)
#
#     def retrieve_trimming_curves(self, surface_index):
#         """
#         Extract trimming curves for a given surface from the topology data.
#
#         Args:
#         surface_index (int): Index of the surface.
#
#         Returns:
#         list: List of trimming curves for the specified surface.
#         """
#         assert 0 <= surface_index < len(self._surfaces), "Invalid surface_index"
#
#         trimming_curves = []
#         for t in self._topology:
#             for solid in t.solids:
#                 for shell_index in solid['shells']:
#                     shell = t.shells[shell_index]
#                     for face_info in shell['faces']:
#                         face = t.faces[face_info['face_index']]
#                         if face['surface'] == surface_index:
#                             for loop_id in face['loops']:
#                                 loop = t.loops[loop_id]
#                                 for he in loop['halfedges']:
#                                     half_edge = t.halfedges[he]
#                                     edge = t.edges[half_edge['edge']]
#                                     curve = self._curves3d[edge['3dcurve']]
#                                     trimming_curves.append(curve)
#
#         return trimming_curves
#
#     def calculate_winding_numbers(self, sampled_points, trimming_curves):
#         """
#         Determine winding numbers for sampled points to identify points inside the trimmed region.
#
#         Args:
#         sampled_points (numpy.ndarray): Sampled points on the surface.
#         trimming_curves (list): List of trimming curves.
#
#         Returns:
#         numpy.ndarray: Winding numbers for each sampled point.
#         """
#         winding_numbers = np.zeros(sampled_points.shape[0])
#         for curve in trimming_curves:
#             # Assuming `calculate_winding_number_for_curve` is a utility function that
#             # computes winding numbers for a single curve
#             winding_numbers += calculate_winding_number_for_curve(sampled_points, curve)
#
#         return winding_numbers
#
#     # def calculate_winding_numbers(self, surface_index, sampled_points):
#     #     # Implementation of winding number calculation
#     #     # ...
#     #     trimming_curves = self.retrieve_trimming_curves(surface_index)
#     #     winding_numbers = np.zeros(len(sampled_points))
#     #
#     #     for curve in trimming_curves:
#     #         winding_numbers += self._winding_number_calculator(sampled_points, curve)
#     #
#     #     return winding_numbers
#     #
#     # def _winding_number_calculator(self, points, curve):
#     #     # This method would use the optimized winding number calculation method
#     #     # For example, using the optimized_winding_number function
#     #     # ...
#     #     return winding_numbers
#
#     def calculate_surface_area(self):
#         """
#         Compute the area of each surface in the shape.
#
#         Returns:
#         list: List of surface areas.
#         """
#         surface_areas = []
#         for surface in self._surfaces:
#             surface_area = self._calculate_individual_surface_area(surface)
#             surface_areas.append(surface_area)
#
#         return surface_areas
#
#     def _calculate_individual_surface_area(self, surface):
#         """
#         Calculate the area of an individual surface.
#
#         Args:
#         surface: The surface object.
#
#         Returns:
#         float: The calculated area of the surface.
#         """
#         # Sample points uniformly across the surface
#         sample_points = uniform_points(surface, SURFACE_POINTS)  # Example, using 100 sample points
#
#         # Compute the first fundamental form
#         total_area = 0
#         for point in sample_points:
#             E, F, G = self._first_fundamental_form(surface, point)
#             area_element = np.sqrt(E * G - F ** 2)
#             total_area += area_element
#
#         total_area /= len(sample_points)
#         return total_area
#
#     def _first_fundamental_form(self, surface, point):
#         """
#         Compute the components of the first fundamental form at a point on the surface.
#
#         Args:
#         surface: The surface object.
#         point: The point on the surface.
#
#         Returns:
#         tuple: Components E, F, G of the first fundamental form.
#         """
#         # Derivatives of the parametric surface representation
#         dS_du, dS_dv = surface.derivative(point, 1)
#
#         # Components of the first fundamental form
#         E = np.dot(dS_du, dS_du)
#         F = np.dot(dS_du, dS_dv)
#         G = np.dot(dS_dv, dS_dv)
#
#         return E, F, G
#
#     def get_surface_properties(self, surface_index):
#         """
#         Get properties of a surface by its index.
#
#         Args:
#         surface_index (int): The index of the surface.
#
#         Returns:
#         dict: Properties of the surface.
#         """
#         surface = self.get_surface_by_index(surface_index)
#         return {
#             'type': surface._type,
#             'trim_domain': surface._trim_domain,
#             'area_size': self._area_size[surface_index]
#         }
#
#     def sample_all_surfaces(self, spacing=1e-1):
#         """
#         Sample points on all surfaces.
#
#         Args:
#         spacing (float): Spacing parameter for sampling.
#
#         Returns:
#         dict: Sampled points for each surface.
#         """
#         sampled_points = {}
#         for i in range(len(self._surfaces)):
#             sampled_points[i] = self.sample_surface(i, spacing)
#         return sampled_points
#
#     def sample_all_curves(self, spacing=1e-1, is_3d=True):
#         """
#         Sample points on all curves.
#
#         Args:
#         spacing (float): Spacing parameter for sampling.
#         is_3d (bool): Flag to sample 3D or 2D curves.
#
#         Returns:
#         dict: Sampled points for each curve.
#         """
#         sampled_points = {}
#         curves = self._curves3d if is_3d else self._curves2d
#         for i, curve in enumerate(curves):
#             samples = self._generate_curve_samples(curve, spacing)
#             sampled_points[i] = curve.sample(samples)
#         return sampled_points
#
#     def get_curve_derivative(self, curve_index, sample_points, order, is_3d=True):
#         """
#         Get the derivative of a curve at given points.
#
#         Args:
#         curve_index (int): The index of the curve.
#         sample_points (np.ndarray): Points to sample the derivative.
#         order (int): Order of the derivative.
#         is_3d (bool): Flag for 3D or 2D curve.
#
#         Returns:
#         np.ndarray: Derivative values at the sample points.
#         """
#         curve = self.get_curve_by_index(curve_index, is_3d)
#         return curve.derivative(sample_points, order)
#
#     def get_surface_normal(self, surface_index, sample_points):
#         """
#         Get the normal vectors of a surface at given points.
#
#         Args:
#         surface_index (int): The index of the surface.
#         sample_points (np.ndarray): Points on the surface to sample the normals.
#
#         Returns:
#         np.ndarray: Normal vectors at the sample points.
#         """
#         surface = self.get_surface_by_index(surface_index)
#         return surface.normal(sample_points)
#
#     # Private helper methods
#     def _generate_curve_samples(self, curve, spacing):
#         """
#         Generate sample points for a curve based on the given spacing.
#
#         Args:
#         curve: Curve object.
#         spacing (float): Spacing for sampling.
#
#         Returns:
#         np.ndarray: Generated sample points.
#         """
#         min_i, max_i = curve._interval
#         sample_count = int((max_i - min_i) / spacing)
#         return np.linspace(min_i, max_i, sample_count)
