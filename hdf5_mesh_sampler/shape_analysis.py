# shape_analysis.py
import numpy as np
from shape_core import ShapeCore

class ShapeAnalysis(ShapeCore):

    def __init__(self, shape_core_instance):
        if not isinstance(shape_core_instance, ShapeCore):
            raise ValueError("Expected an instance of ShapeCore")
        # Initialize with existing base class properties
        super().__init__({}, {})
        self._curves2d = shape_core_instance._curves2d
        self._curves3d = shape_core_instance._curves3d
        self._surfaces = shape_core_instance._surfaces
        self._bbox = shape_core_instance._bbox
        self._topology = shape_core_instance._topology
    def calculate_surface_area(self):
        """
        Compute the area of each surface in the shape.

        Returns:
        list: List of surface areas.
        """
        surface_areas = []
        for surface in self._surfaces:
            surface_area = self._calculate_individual_surface_area(surface)
            surface_areas.append(surface_area)

        return surface_areas

    def _calculate_individual_surface_area(self, surface):
        """
        Calculate the area of an individual surface.

        Args:
        surface: The surface object.

        Returns:
        float: The calculated area of the surface.
        """
        # Example implementation, details depend on your surface representation
        sample_points = self._generate_uniform_points(surface)  # Generate uniform points on the surface
        total_area = 0
        for point in sample_points:
            E, F, G = self._first_fundamental_form(surface, point)
            area_element = np.sqrt(E * G - F ** 2)
            total_area += area_element

        return total_area / len(sample_points)

    def _first_fundamental_form(self, surface, point):
        """
        Compute the components of the first fundamental form at a point on the surface.

        Args:
        surface: The surface object.
        point: The point on the surface.

        Returns:
        tuple: Components E, F, G of the first fundamental form.
        """
        # Example implementation, details depend on your surface representation
        dS_du, dS_dv = surface.derivative(point, 1)
        E = np.dot(dS_du, dS_du)
        F = np.dot(dS_du, dS_dv)
        G = np.dot(dS_dv, dS_dv)

        return E, F, G

    def get_surface_normal(self, surface_index, sample_points):
        """
        Get the normal vectors of a surface at given points.

        Args:
        surface_index (int): The index of the surface.
        sample_points (np.ndarray): Points on the surface to sample the normals.

        Returns:
        np.ndarray: Normal vectors at the sample points.
        """
        assert 0 <= surface_index < len(self._surfaces), "Invalid surface_index"
        surface = self._surfaces[surface_index]
        return np.array([surface.normal(p) for p in sample_points])
