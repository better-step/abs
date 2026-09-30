"""
Computations for winding number and related operations for curves/surfaces.
"""

import numpy as np

def winding_number(curve_uv_values, surface_uv_values, chunk_size=20000):
        av = curve_uv_values[:-1]
        bv = curve_uv_values[1:]
        n = len(surface_uv_values)
        out = np.empty(n, dtype=np.float64)

        ax0, ax1 = av[:, 0], av[:, 1]
        bx0, bx1 = bv[:, 0], bv[:, 1]

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            p = surface_uv_values[start:end]
            px = p[:, 0][None, :]
            py = p[:, 1][None, :]

            ax = ax0[:, None] - px
            ay = ax1[:, None] - py
            bx = bx0[:, None] - px
            by = bx1[:, None] - py

            det = ax * by - ay * bx
            dot = ax * bx + ay * by

            out[start:end] = np.sum(np.arctan2(det, dot), axis=0) / (2 * np.pi)

        return out[:, None]

def find_surface_uv_for_curve(surface_points, surface_uv_values, curve_points):
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
        # nearest_3d_surface_points, curve_indexes = self._calculate_nearest_surface_points(surface_points, curve_points)
        from scipy.spatial import KDTree
        tree = KDTree(surface_points)
        _, curve_indexes = tree.query(curve_points)

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
