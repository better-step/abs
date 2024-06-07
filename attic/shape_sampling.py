# shape_sampling.py
import numpy as np
from shape_core import ShapeCore
from winding_number.winding_number import calculate_winding_numbers


class ShapeSampling(ShapeCore):
    def sample_curve(self, curve_index, sampler):
        """
        Sample points on a 3D curve.

        Args:
        curve_index (int): Index of the curve to be sampled.
        sampler (CurveSampler): An instance of a CurveSampler.

        Returns:
        numpy.ndarray: Array of sampled points on the curve.
        """
        assert 0 <= curve_index < len(self._curves3d), "Invalid curve_index"
        curve = self._curves3d[curve_index]
        return sampler.sample(curve)

    def sample_all_shapes(self, surface_sampler, curve_sampler):
        """
        Sample all shapes in the model.

        Args:
        surface_sampler (SurfaceSampler): An instance of SurfaceSampler.
        curve_sampler (CurveSampler): An instance of CurveSampler.

        Returns:
        dict: Sampled points for each surface with key as surface index.
        """
        sampled_shapes = {}

        for part_index, part in enumerate(self._topology):
            part_shapes = {}
            for solid_index, solid in enumerate(part.solids):
                for shell_index in solid['shells']:
                    shell = part.shells[shell_index]
                    for (face_index,_) in shell['faces']:
                        face = part.faces[face_index]
                        sampled_surface_points = self.sample_single_surface(part_index,face, surface_sampler, curve_sampler)
                        if sampled_surface_points.size == 0:
                            part_shapes[face['surface']] = None
                            continue
                        part_shapes[face['surface']] = sampled_surface_points
            sampled_shapes[part_index] = part_shapes
        return sampled_shapes

    def sample_single_surface(self, part_index, face, surface_sampler, curve_sampler):
        """
        Sample a single surface based on face information, considering trimming curves.

        Args:
            part (Part): The part object containing the surface.
            face (dict): Information about the face to be sampled.

        Returns:
            numpy.ndarray: Array of sampled points on the surface.
        """
        surface_index = face['surface']
        surface = self._surfaces[surface_index]  # Todo: Handle for different parts

        # Sample the surface to get UV values and corresponding 3D points
        surface_uv_values = surface_sampler.sample(surface)
        surface_points = surface.sample(surface_uv_values)

        print("surface_points shape: " + str(surface_points.shape))

        if surface_points.size == 0:
            print("surface_points size is 0" + str(surface_index))
            return surface_points

        total_winding_numbers = np.zeros(len(surface_uv_values))

        # Process trimming curves
        for loop_id in face['loops']:
            loop = self._topology[part_index].loops[loop_id]
            for halfedge_index in loop['halfedges']:
                halfedge = self._topology[part_index].halfedges[halfedge_index]
                curve3d_index = halfedge['edge']
                curve = self._curves3d[curve3d_index]
                modified_orientation = self.determine_curve_orientation(face, halfedge)

                # Sample the curve points to get UV values
                curve_uv_points = curve_sampler.sample(curve)

                # Convert the curve points to 3D points
                curve_points = curve.sample(curve_uv_points)

                if not modified_orientation:
                    curve_points = curve_points[::-1]

                # Calculate the nearest UV values on the surface for the curve points
                closest_surface_uv_values_of_curve = self.find_surface_uv_for_curve(surface_points, surface_uv_values,
                                                                                    curve_points)

                # Determine the periodicity of the surface
                period_u, period_v = self._determine_surface_periodicity(surface)

                if closest_surface_uv_values_of_curve.size == 0:
                    print("closest_surface_uv_values_of_curve size is 0" + str(curve3d_index))
                    continue

                # Calculate the winding number for the curve
                wn = calculate_winding_numbers(closest_surface_uv_values_of_curve, surface_uv_values, period_u,
                                                   period_v)

                if any(wn > 0.5):
                    print("curve contains points on the surface" + str(curve3d_index))

                total_winding_numbers += wn.squeeze()

        # Filter points based on total winding number
        final_points = surface_points[total_winding_numbers > 0.5]

        return final_points

    def determine_curve_orientation(self, face, halfedge):
        """
        Determine the orientation of a curve relative to a surface.

        Args:
            face (dict): The face information.
            halfedge (dict): The halfedge information.

        Returns:
            bool: The modified orientation of the curve.
        """
        orientation_wrt_edge = halfedge['orientation_wrt_edge']
        if not face['surface_orientation']:
            orientation_wrt_edge = not orientation_wrt_edge
        return orientation_wrt_edge

    def find_surface_uv_for_curve(self, surface_points, surface_uv_values, curve_points):
        """
        Calculate the nearest UV values on a surface for a given set of curve points.

        Args:
        surface_points (np.ndarray): Points on the surface.
        surface_uv_values (np.ndarray): UV values on the surface.
        curve_points (np.ndarray): Points on the curve.

        Returns:
        np.ndarray: UV values on the surface closest to the curve points.
        """
        # Calculate the nearest surface point for each curve point
        nearest_3d_surface_points, curve_indexes = self._calculate_nearest_surface_points(surface_points, curve_points)

        if type(curve_indexes) == np.int64:
            curve_indexes = [curve_indexes]

        if surface_uv_values.size > 0 and max(curve_indexes) < len(surface_uv_values):
            surface_uv_near_curve = surface_uv_values[curve_indexes]
        else:
            # Handle the case where surface_uv_values is empty or too small
            # This could be setting surface_uv_near_curve to an empty array
            # or some other default value, depending on your application's needs
            surface_uv_near_curve = np.array([])

        return surface_uv_near_curve

    def _calculate_nearest_surface_points(self, surface_points, curve_points):
        """
        Calculate the nearest surface points for a given set of curve points using a KDTree.

        Args:
        surface_points (np.ndarray): Points on the surface.
        curve_points (np.ndarray): Points on the curve.

        Returns:
        np.ndarray: Surface points closest to the curve points.
        """
        from scipy.spatial import KDTree
        tree = KDTree(surface_points)
        _, nearest_surface_point_indices = tree.query(curve_points)
        return surface_points[nearest_surface_point_indices], nearest_surface_point_indices

    def _determine_surface_periodicity(self, surface):
        """
        Determine the periodicity of a surface based on its type and properties.
        """
        flatten_trim_domain = surface._trim_domain.flatten()
        if surface._type in ["Plane", "BSpline", "Extrusion", "Other"]:
            return None, None
        elif surface._type in ["Cylinder", "Cone", "Revolution"]:
            return flatten_trim_domain[1] - flatten_trim_domain[0], None
        elif surface._type in ["Sphere", "Torus"]:
            return flatten_trim_domain[1] - flatten_trim_domain[0], flatten_trim_domain[3] - flatten_trim_domain[2]
        else:
            return None, None  # Default case for other types

    # def sample_single_surface_old(self, surface_index, surface_sampler, curve_sampler):
    #     """
    #     Sample a single surface in the model.
    #
    #     Args:
    #     surface_index (int): Index of the surface to be sampled.
    #     surface_sampler (SurfaceSampler): An instance of SurfaceSampler.
    #     curve_sampler (CurveSampler): An instance of CurveSampler.
    #
    #     Returns:
    #     numpy.ndarray: Array of sampled points on the surface.
    #     """
    #     # Validate surface_index
    #     if not 0 <= surface_index < len(self._surfaces):
    #         raise ValueError(f"Invalid surface_index: {surface_index}")
    #
    #     # Retrieve the surface to be sampled
    #     surface = self._surfaces[surface_index]
    #
    #     # Sample the surface to get UV values and corresponding 3D points
    #     surface_uv_values = surface_sampler.sample(surface)
    #     surface_points = surface.sample(surface_uv_values)
    #
    #     # Initialize winding number for each UV value
    #     total_winding_numbers = np.zeros(surface_uv_values.shape[0])
    #
    #     # Process each curve related to the surface
    #     trimming_curves = self.retrieve_trimming_curves(surface_index)
    #
    #     if len(trimming_curves) == 0:
    #         return surface_points
    #
    #
    #     for curve, modified_orientation in trimming_curves:
    #         # Sample the curve points to get UV values
    #         curve_uv_points = curve_sampler.sample(curve)
    #
    #         # Convert the curve points to 3D points
    #         curve_points = curve.sample(curve_uv_points)
    #
    #         # Calculate the nearest UV values on the surface for the curve points
    #         closest_surface_uv_values_of_curve = self.find_surface_uv_for_curve(surface_points, surface_uv_values, curve_points)
    #
    #         if not modified_orientation:  # Surface orientation is opposite to curve orientation
    #             # Reverse the curve points
    #             closest_surface_uv_values_of_curve = closest_surface_uv_values_of_curve[::-1]
    #
    #         # Determine the periodicity of the surface
    #         period_u, period_v = self._determine_surface_periodicity(surface)
    #
    #         # Calculate the winding number for the curve
    #         wn = calculate_winding_numbers(closest_surface_uv_values_of_curve, surface_uv_values, period_u, period_v)
    #
    #         # Add winding numbers from this curve to the total
    #         total_winding_numbers += wn.squeeze()
    #
    #     # Filter points based on total winding number
    #     final_points = surface_points[total_winding_numbers > 0.5]
    #
    #
    #
    #     return final_points
    # def retrieve_trimming_curves(self, surface_index):
    #     """
    #     Extract trimming curves and their orientations for a given surface from the topology data.
    #
    #     Args:
    #     surface_index (int): Index of the surface.
    #
    #     Returns:
    #     list: List of tuples, each containing a trimming curve and its orientation (orientation_wrt_edge) for the specified surface.
    #     """
    #     assert 0 <= surface_index < len(self._surfaces), "Invalid surface_index"
    #
    #     trimming_curves = []
    #     for t in self._topology:
    #         for solid in t.solids:
    #             for shell_index in solid['shells']:
    #                 shell = t.shells[shell_index]
    #                 for face_info in shell['faces']:
    #                     face = t.faces[face_info['face_index']]
    #                     # got the surface and its sampled points
    #                     # initilized polygons
    #                     if face['surface'] == surface_index:
    #                         surface_orientation = face['surface_orientation']
    #                         for loop_id in face['loops']:
    #                             loop = t.loops[loop_id]
    #                             for he in loop['halfedges']:
    #                                 half_edge = t.halfedges[he]
    #                                 edge = t.edges[half_edge['edge']]
    #                                 curve = self._curves3d[edge['3dcurve']]
    #                                 # got the curve and its sampled points
    #                                 orientation_wrt_edge = half_edge['orientation_wrt_edge']
    #                                 if not surface_orientation:
    #                                     orientation_wrt_edge = not orientation_wrt_edge
    #                                 trimming_curves.append((curve, orientation_wrt_edge))
    #
    #     return trimming_curves
