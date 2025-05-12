"""
Curve classes for parametric curves (Line, Circle, Ellipse, B-spline, etc).
Provide methods to sample points on the curve and compute derivatives.
"""
import numpy as np
from scipy.interpolate import BSpline
from geomdl import NURBS


def create_curve(curve_data, compute_index=True):
    """
    Factory function to create a Curve object from HDF5 curve data.
    Returns (index, curve_object). If compute_index is False, index is None.
    """
    if compute_index:
        index = int(curve_data.name.split("/")[-1])
    else:
        index = None
    curve_type = curve_data.get('type')[()].decode('utf-8')
    curve_map = {
        'Line': Line,
        'Circle': Circle,
        'Ellipse': Ellipse,
        'BSpline': BSplineCurve,
        'Other': Other
    }
    curve_class = curve_map.get(curve_type)
    if curve_class:
        return index, curve_class(curve_data)
    else:
        # Unsupported curve type
        return index, None


class Curve:
    """Base class for all curve types."""
    def sample(self, points):
        """Sample points on the curve at given parametric values."""
        raise NotImplementedError("Sample method must be implemented by subclasses")
    def derivative(self, points, order=1):
        """Evaluate the curve's derivative of given order at parametric points."""
        raise NotImplementedError("Derivative method must be implemented by subclasses")
    def get_length(self):
        """Approximate the curve length via Gaussian quadrature (4-point Legendre)."""
        if getattr(self, "length", None) is not None and self.length != -1:
            return self.length
        # Compute length via integration of first derivative
        pts, weights = np.polynomial.legendre.leggauss(4)
        pts = (pts + 1) * 0.5 * (self.interval[0, 1] - self.interval[0, 0]) + self.interval[0, 0]
        deriv = self.derivative(pts[:, None], order=1)
        lengths = np.linalg.norm(deriv, axis=1)
        self.length = np.sum(lengths * weights) * (self.interval[0, 1] - self.interval[0, 0]) / 2
        return self.length



class Line(Curve):
    """Straight line curve defined by a start location and direction vector."""
    def __init__(self, line):
        self.location = np.array(line.get('location')[()]).reshape(-1, 1).T
        self.interval = np.array(line.get('interval')[()]).reshape(-1, 1).T
        self.direction = np.array(line.get('direction')[()]).reshape(-1, 1).T
        if self.direction.shape[1] == 3:
            self.transform = np.array(line.get('transform')[()])  # transformation matrix if present
        self.length = -1
        self.shape_name = line.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self.location
        # Parametric coordinate in interval [0,1] maps to location + t * direction
        return self.location + sample_points * self.direction

    def get_length(self):
        if self.length == -1:
            self.length = np.linalg.norm((self.interval[0, 1] - self.interval[0, 0]) * self.direction)
        return self.length

    def derivative(self, sample_points, order=1):
        if order == 1:
            # First derivative of line is constant direction
            return np.tile(self.direction, (sample_points.shape[0], 1))
        # Higher derivatives of a line are zero
        return np.zeros((sample_points.shape[0], self.location.shape[1]))


class Circle(Curve):
    """Circular arc or full circle curve."""
    def __init__(self, circle):
        self.location = np.array(circle.get('location')[()]).reshape(-1, 1).T
        self.radius = float(circle.get('radius')[()])
        self.interval = np.array(circle.get('interval')[()]).reshape(-1, 1).T
        self.x_axis = np.array(circle.get('x_axis')[()]).reshape(-1, 1).T
        self.y_axis = np.array(circle.get('y_axis')[()]).reshape(-1, 1).T
        self.length = -1
        self.shape_name = circle.get('type')[()].decode('utf8')
    def sample(self, sample_points):
        if sample_points.size == 0:
            return self.location
        # parametric angle values in sample_points
        circle_points = self.location + self.radius * (np.cos(sample_points) * self.x_axis + np.sin(sample_points) * self.y_axis)
        return circle_points
    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        # Circle derivatives are periodic every 4th derivative
        if order % 4 == 0 and order != 0:
            # Cosine form (same as first derivative for order 4)
            return self.radius * (np.cos(sample_points) * self.x_axis + np.sin(sample_points) * self.y_axis)
        elif order % 4 == 1:
            return -self.radius * np.sin(sample_points) * self.x_axis + self.radius * np.cos(sample_points) * self.y_axis
        elif order % 4 == 2:
            return -self.radius * (np.cos(sample_points) * self.x_axis + np.sin(sample_points) * self.y_axis)
        else:  # order % 4 == 3
            return self.radius * (np.sin(sample_points) * self.x_axis - np.cos(sample_points) * self.y_axis)


class Ellipse(Curve):
    """Elliptical curve (full ellipse or arc)."""
    def __init__(self, ellipse):
        self.focus1 = np.array(ellipse.get('focus1')[()]).reshape(-1, 1).T
        self.focus2 = np.array(ellipse.get('focus2')[()]).reshape(-1, 1).T
        self.interval = np.array(ellipse.get('interval')[()]).reshape(-1, 1).T
        self.maj_radius = float(ellipse.get('maj_radius')[()])
        self.min_radius = float(ellipse.get('min_radius')[()])
        self.x_axis = np.array(ellipse.get('x_axis')[()]).reshape(-1, 1).T
        self.y_axis = np.array(ellipse.get('y_axis')[()]).reshape(-1, 1).T
        self.length = -1
        self.shape_name = ellipse.get('type')[()].decode('utf8')
        if 'z_axis' in ellipse:
            self.z_axis = np.array(ellipse.get('z_axis')[()]).reshape(-1, 1).T
            self.transform = np.array(ellipse.get('transform')[()])  # 4x4 transform matrix if present
        # Center of ellipse
        self.center = (self.focus1 + self.focus2) / 2
    def sample(self, sample_points):
        # Parametric angle values in sample_points
        ellipse_points = self.center + self.maj_radius * np.cos(sample_points) * self.x_axis + self.min_radius * np.sin(sample_points) * self.y_axis
        return ellipse_points
    def derivative(self, sample_points, order=1):
        if order % 4 == 0:
            if order == 0:
                return self.sample(sample_points)
            return self.maj_radius * np.cos(sample_points) * self.x_axis + self.min_radius * np.sin(sample_points) * self.y_axis
        elif order % 4 == 1:
            return -self.maj_radius * np.sin(sample_points) * self.x_axis + self.min_radius * np.cos(sample_points) * self.y_axis
        elif order % 4 == 2:
            return -self.maj_radius * np.cos(sample_points) * self.x_axis - self.min_radius * np.sin(sample_points) * self.y_axis
        else:
            return self.maj_radius * np.sin(sample_points) * self.x_axis - self.min_radius * np.cos(sample_points) * self.y_axis


class BSplineCurve(Curve):
    """B-spline or NURBS curve."""
    def __init__(self, bspline):
        self.length = -1
        self.closed = bool(bspline.get('closed')[()])
        self.degree = int(bspline.get('degree')[()])
        self.continuity = int(bspline.get('continuity')[()])
        self.poles = np.array(bspline.get('poles')[()])  # control points array
        self.knots = np.array(bspline.get('knots')[()]).reshape(-1, 1).T
        self.weights = np.array(bspline.get('weights')[()]).reshape(-1, 1)
        self.interval = np.array(bspline.get('interval')[()]).reshape(-1, 1).T
        self.rational = bool(bspline.get('rational')[()])
        self.periodic = bool(bspline.get('periodic')[()])
        self.shape_name = bspline.get('type')[()].decode('utf8')
        if self.poles.shape[1] == 3:
            self.transform = np.array(bspline.get('transform')[()])  # apply if present
        # Create underlying BSpline or NURBS curve object
        if self.rational:
            # Use NURBS-Python for rational B-spline
            self.bspline = NURBS.Curve()  # no normalization of knots to allow exact usage
            self.bspline.degree = self.degree
            self.bspline.ctrlpts = self.poles.tolist()
            self.bspline.knotvector = self.knots.flatten().tolist()
            self.bspline.weights = self.weights.flatten().tolist()
        else:
            # Use SciPy BSpline for non-rational
            self.bspline = BSpline(self.knots.T[0], self.poles, self.degree)
    def sample(self, sample_points):
        if sample_points.size == 0:
            return np.array([]).reshape(0, self.poles.shape[1])
        if self.rational:
            # Evaluate rational B-spline (list of points)
            return np.array(self.bspline.evaluate_list(sample_points.flatten().tolist()))
        # Evaluate SciPy BSpline (returns (n_points, dim) array)
        return np.squeeze(self.bspline(sample_points))
    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        # If requested derivative order is higher than degree, derivative is zero vector
        if self.degree < order:
            return np.zeros((sample_points.shape[0], self.poles.shape[1]))
        if self.rational:
            # Use geomdl to compute derivatives for rational curves
            res = np.zeros((sample_points.shape[0], self.poles.shape[1]))
            for i in range(sample_points.shape[0]):
                d = self.bspline.derivatives(sample_points[i, 0], order)
                res[i, :] = d[-1]  # d[-1] is highest derivative (order-th)
            return res
        # For non-rational, SciPy's BSpline object can provide derivative by constructing a new BSpline
        b_spline_deriv = self.bspline.derivative(nu=order)
        return np.squeeze(b_spline_deriv(sample_points))

class Other(Curve):
    """Fallback for unsupported curve types."""
    def __init__(self, other):
        self.shape_name = other.get('type')[()].decode('utf8')
        self.interval = np.array(other.get('interval')[()]).reshape(-1, 1).T
    def sample(self, sample_points):
        # Unsupported: return empty array
        return np.array([]).reshape(0, 3)
    def derivative(self, sample_points, order=1):
        return np.array([]).reshape(0, 3)
    def get_length(self):
        return 0.0


