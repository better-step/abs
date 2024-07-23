import numpy as np

def compute_position(point, period_u, period_v):
    """ Compute the 3D position of a point from its UV coordinates and periods. """
    u, v = point[:2]
    position = np.zeros(3)

    if period_u == 0 or period_v == 0:
       # print("Warning: Periodicity is 0, returning 0 vector.")
        return position

    if period_u is not None and period_v is not None:
        position[0] = np.cos(u / period_u * 2 * np.pi) * np.cos(v / period_v * 2 * np.pi)
        position[1] = np.cos(u / period_u * 2 * np.pi) * np.sin(v / period_v * 2 * np.pi)
        position[2] = np.sin(u / period_u * 2 * np.pi)
    elif period_u is not None:
        position[0] = np.sin(u / period_u * 2 * np.pi)
        position[1] = np.cos(u / period_u * 2 * np.pi)
        position[2] = v
    elif period_v is not None:
        position[0] = u
        position[1] = np.sin(v / period_v * 2 * np.pi)
        position[2] = np.cos(v / period_v * 2 * np.pi)
    else:
        position = np.array(point)
    return position

def optimized_winding_number(curve_uv_values, surface_uv_values, period_u=None, period_v=None):
    """ Compute the winding number for a polyline and surface UV values efficiently. """
    polyline_positions = np.array([compute_position(point, period_u, period_v) for point in curve_uv_values])
    is_2D = polyline_positions.shape[1] == 2
    a_values = polyline_positions[:-1]
    b_values = polyline_positions[1:]
    positions = np.array([compute_position(point, period_u, period_v) for point in surface_uv_values])
    a = a_values[:, np.newaxis] - positions
    b = b_values[:, np.newaxis] - positions

    if is_2D:
        det = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    else:  # 3D
        det = np.linalg.norm(np.cross(a, b), axis=-1)

    dot = np.einsum('ijk,ijk->ij', a, b)
    winding_number_result = np.sum(np.arctan2(det, dot), axis=0) / (2 * np.pi)
    return winding_number_result.reshape(-1, 1)

def calculate_winding_numbers(curve_uv_values_on_surface, surface_uv_values, period_u=None, period_v=None):
    """ Public API function to calculate winding numbers for a given set of surface points. """


    # input for winding number calculation -> curve_uv_values, surface_uv_values, period_u, period_v



    return optimized_winding_number(curve_uv_values_on_surface, surface_uv_values, period_u, period_v)


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