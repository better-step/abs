# shape_sampling.py
import numpy as np
from shape_core import ShapeCore

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

    def sample_surface(self, surface_index, sampler, winding_number_calculator):
        """
        Sample points on a surface, considering trimming curves and winding numbers.

        Args:
        surface_index (int): Index of the surface to be sampled.
        sampler (SurfaceSampler): An instance of a SurfaceSampler.
        winding_number_calculator (function): A function to calculate winding numbers.

        Returns:
        numpy.ndarray: Array of sampled points on the surface.
        """
        assert 0 <= surface_index < len(self._surfaces), "Invalid surface_index"
        surface = self._surfaces[surface_index]

        # Sample points on the surface
        sampled_points = sampler.sample(surface)

        # Filter points based on winding number
        filtered_points = []
        for point in sampled_points:
            wn = winding_number_calculator(point, self._trimming_curves[surface_index])
            if wn > 0.5:  # Assuming the winding number threshold is 0.5
                filtered_points.append(point)

        return np.array(filtered_points)

    def retrieve_trimming_curves(self, surface_index):
        """
        Extract trimming curves for a given surface from the topology data.

        Args:
        surface_index (int): Index of the surface.

        Returns:
        list: List of trimming curves for the specified surface.
        """
        assert 0 <= surface_index < len(self._surfaces), "Invalid surface_index"

        trimming_curves = []
        for t in self._topology:
            for solid in t.solids:
                for shell_index in solid['shells']:
                    shell = t.shells[shell_index]
                    for face_info in shell['faces']:
                        face = t.faces[face_info['face_index']]
                        if face['surface'] == surface_index:
                            for loop_id in face['loops']:
                                loop = t.loops[loop_id]
                                for he in loop['halfedges']:
                                    half_edge = t.halfedges[he]
                                    edge = t.edges[half_edge['edge']]
                                    curve = self._curves3d[edge['3dcurve']]
                                    trimming_curves.append(curve)

        return trimming_curves
