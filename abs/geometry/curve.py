import numpy as np
from geomdl import BSpline, NURBS
from scipy.integrate import quad


class Curve:
    def sample(self, points):
        raise NotImplementedError("Sample method must be implemented by subclasses")

    def length(self):
        num_samples = 100
        param_range = np.linspace(self._interval[0, 0], self._interval[0, 1], num_samples)
        points = self.sample(param_range.reshape(-1, 1))
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    def derivative(self, points, order=1):
        raise NotImplementedError("Derivative method must be implemented by subclasses")

    def normal(self, points):
        raise NotImplementedError("Normal method must be implemented by subclasses")


class Line(Curve):
    def __init__(self, line):
        if isinstance(line, dict):
            self._location = np.array(line['location']).reshape(-1, 1).T
            self._interval = np.array(line['interval']).reshape(-1, 1).T
            self._direction = np.array(line['direction']).reshape(-1, 1).T
            self._type = line['type']
        else:
            self._location = np.array(line.get('location')[()]).reshape(-1, 1).T
            self._interval = np.array(line.get('interval')[()]).reshape(-1, 1).T
            self._direction = np.array(line.get('direction')[()]).reshape(-1, 1).T
            self._type = line.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self._location
        return self._location + sample_points * self._direction

    def length(self):
        return np.linalg.norm((self._interval[0, 1] - self._interval[0, 0]) * self._direction)

    def derivative(self, sample_points, order=1):
        if order == 1:
            return np.tile(self._direction, (sample_points.shape[0], 1))
        return np.zeros([sample_points.shape[0], self._location.shape[1]])

    def normal(self, sample_points):
        # Assuming the direction is (dx, dy), a perpendicular direction can be (dy, -dx) or (-dy, dx).
        normal_vector = np.array([-self._direction[0, 1], self._direction[0, 0]])
        return np.tile(normal_vector, (sample_points.shape[0], 1))


class Circle(Curve):
    def __init__(self, circle):
        if isinstance(circle, dict):
            self._location = np.array(circle['location']).reshape(-1, 1).T
            self._radius = float(circle['radius'])
            self._interval = np.array(circle['interval']).reshape(-1, 1).T
            self._x_axis = np.array(circle['x_axis']).reshape(-1, 1).T
            self._y_axis = np.array(circle['y_axis']).reshape(-1, 1).T
            self._type = circle['type']
            if 'z_axis' in circle:
                self._z_axis = np.array(circle['z_axis']).reshape(-1, 1).T
        else:
            self._location = np.array(circle.get('location')[()]).reshape(-1, 1).T
            self._radius = float(circle.get('radius')[()])
            self._interval = np.array(circle.get('interval')[()]).reshape(-1, 1).T
            self._x_axis = np.array(circle.get('x_axis')[()]).reshape(-1, 1).T
            self._y_axis = np.array(circle.get('y_axis')[()]).reshape(-1, 1).T
            self._type = circle.get('type')[()].decode('utf8')
            if 'z_axis' in circle:
                self._z_axis = np.array(circle.get('z_axis')[()]).reshape(-1, 1).T

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self._location
        circle_points = self._location + self._radius * (
            np.cos(sample_points) * self._x_axis + np.sin(sample_points) * self._y_axis)
        return circle_points

    def length(self):
        # Circumference of the circle
        norm_x = np.linalg.norm(self._x_axis)
        norm_y = np.linalg.norm(self._y_axis)
        integrand = lambda t: np.sqrt(
            (-self._radius * np.sin(t) * norm_x) ** 2 + (self._radius * np.cos(t) * norm_y) ** 2)
        circumference, _ = quad(integrand, self._interval[0, 0], self._interval[0, 1])
        return circumference

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order % 4 == 0 and order != 0:
            return self._radius * (np.cos(sample_points) * self._x_axis + np.sin(sample_points) * self._y_axis)
        elif order % 4 == 1:
            return (-self._radius * np.sin(sample_points) * self._x_axis) + (
                self._radius * np.cos(sample_points) * self._y_axis)
        elif order % 4 == 2:
            return -(self._radius * (np.cos(sample_points) * self._x_axis + np.sin(sample_points) * self._y_axis))
        elif order % 4 == 3:
            return self._radius * (np.sin(sample_points) * self._x_axis - np.cos(sample_points) * self._y_axis)

    def normal(self, sample_points):
        # Normals are radially outward, computed as the difference from the center to the sample points.
        # FIXME: This is not the correct normal calculation for a circle.
        circle_points = self.sample(sample_points) - self._location
        norms = np.linalg.norm(circle_points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return circle_points / norms


class Ellipse(Curve):
    def __init__(self, ellipse):
        if isinstance(ellipse, dict):
            self._focus1 = np.array(ellipse['focus1']).reshape(-1, 1).T
            self._focus2 = np.array(ellipse['focus2']).reshape(-1, 1).T
            self._interval = np.array(ellipse['interval']).reshape(-1, 1).T
            self._maj_radius = float(ellipse['maj_radius'])
            self._min_radius = float(ellipse['min_radius'])
            self._x_axis = np.array(ellipse['x_axis']).reshape(-1, 1).T
            self._y_axis = np.array(ellipse['y_axis']).reshape(-1, 1).T
            self._type = ellipse['type']

            if 'z_axis' in ellipse:
                self._z_axis = np.array(ellipse['z_axis']).reshape(-1, 1).T
        else:
            self._focus1 = np.array(ellipse.get('focus1')[()]).reshape(-1, 1).T
            self._focus2 = np.array(ellipse.get('focus2')[()]).reshape(-1, 1).T
            self._interval = np.array(ellipse.get('interval')[()]).reshape(-1, 1).T
            self._maj_radius = float(ellipse.get('maj_radius')[()])
            self._min_radius = float(ellipse.get('min_radius')[()])
            self._x_axis = np.array(ellipse.get('x_axis')[()]).reshape(-1, 1).T
            self._y_axis = np.array(ellipse.get('y_axis')[()]).reshape(-1, 1).T
            self._type = ellipse.get('type')[()].decode('utf8')

            if 'z_axis' in ellipse:
                self._z_axis = np.array(ellipse.get('z_axis')[()]).reshape(-1, 1).T

        self._center = (self._focus1 + self._focus2) / 2

    def sample(self, sample_points):
        ellipse_points = self._center + self._maj_radius * np.cos(sample_points) * self._x_axis + \
                         self._min_radius * np.sin(sample_points) * self._y_axis
        return ellipse_points

    def length(self):
        # Circumference of the ellipse
        norm_x = np.linalg.norm(self._x_axis)
        norm_y = np.linalg.norm(self._y_axis)
        integrand = lambda t: np.sqrt(
            (-self._maj_radius * np.sin(t) * norm_x) ** 2 + (self._min_radius * np.cos(t) * norm_y) ** 2)
        circumference, _ = quad(integrand, self._interval[0, 0], self._interval[0, 1])
        return circumference

    def derivative(self, sample_points, order=1):
        if order % 4 == 0:
            if order == 0:
                return self.sample(sample_points)
            return self._maj_radius * np.cos(sample_points) * self._x_axis + \
                self._min_radius * np.sin(sample_points) * self._y_axis
        elif order % 4 == 1:
            return -self._maj_radius * np.sin(sample_points) * self._x_axis + \
                self._min_radius * np.cos(sample_points) * self._y_axis
        elif order % 4 == 2:
            return -self._maj_radius * np.cos(sample_points) * self._x_axis - \
                self._min_radius * np.sin(sample_points) * self._y_axis
        return self._maj_radius * np.sin(sample_points) * self._x_axis - \
            self._min_radius * np.cos(sample_points) * self._y_axis

    def normal(self, sample_points):
        # Similar approach as Circle, but considering the ellipse's shape.
        # FIXME: This is not the correct normal calculation for an ellipse.
        ellipse_points = self.sample(sample_points) - self._center
        norms = np.linalg.norm(ellipse_points, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return ellipse_points / norms


class BSplineCurve(Curve):
    def __init__(self, bspline):
        # Attributes initialization for B-spline curve
        if isinstance(bspline, dict):
            self._closed = bool(bspline['closed'])
            self._degree = int(bspline['degree'])
            self._continuity = int(bspline['continuity'])
            self._poles = np.array(bspline['poles'])
            self._knots = np.array(bspline['knots']).reshape(-1, 1).T
            self._weights = np.array(bspline['weights']).reshape(-1, 1)
            self._interval = np.array(bspline['interval']).reshape(-1, 1).T
            self._rational = bool(bspline['rational'])
            self._type = bspline['type']
        else:
            self._closed = bool(bspline.get('closed')[()])
            self._degree = int(bspline.get('degree')[()])
            self._continuity = int(bspline.get('continuity')[()])
            self._poles = np.array(bspline.get('poles')[()])
            self._knots = np.array(bspline.get('knots')[()]).reshape(-1, 1).T
            self._weights = np.array(bspline.get('weights')[()]).reshape(-1, 1)
            self._interval = np.array(bspline.get('interval')[()]).reshape(-1, 1).T
            self._rational = bool(bspline.get('rational')[()])
            self._type = bspline.get('type')[()].decode('utf8')

        # Create BSpline or NURBS curve object
        if self._rational:
            self._curveObject = NURBS.Curve(normalize_kv=False)
        else:
            self._curveObject = BSpline.Curve(normalize_kv=False)

        self._curveObject.degree = self._degree
        self._curveObject.ctrlpts = self._poles.tolist()
        self._curveObject.knotvector = self._knots.flatten().tolist()
        if self._rational:
            self._curveObject.weights = self._weights.flatten().tolist()

    def sample(self, sample_points):
        if sample_points.size == 1:
            return np.array(self._curveObject.evaluate_single(sample_points[0]))

        # Evaluate the curve at the given sample points
        return np.array(self._curveObject.evaluate_list(sample_points[:, 0].tolist()))

    def derivative(self, sample_points, order=1):
        assert (sample_points.shape[1] == 1)
        return np.array([
            self._curveObject.derivatives(sample_points[i, 0], order)[-1] for i in range(sample_points.shape[0])
        ])

    def normal(self, sample_points):
        if sample_points.size == 0:
            return np.array([])

        # Utilize the geomdl built-in normal calculation
        normal_vector = self.derivative(sample_points, order=1)
        normal_vector = np.array([-normal_vector[:, 1], normal_vector[:, 0]]).T

        # Extract just the vector components if normals are returned as tuples (origin, vector)
        return normal_vector
