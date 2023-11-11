import numpy as np
from geomdl import BSpline, NURBS

class Curve:
    def sample(self, points):
        raise NotImplementedError("Sample method must be implemented by subclasses")

    def derivative(self, points, order=1):
        raise NotImplementedError("Derivative method must be implemented by subclasses")


class Line(Curve):
    def __init__(self, line):
        self._location = np.array(line.get('location')[()]).reshape(-1, 1).T
        self._interval = np.array(line.get('interval')[()]).reshape(-1, 1)
        self._direction = np.array(line.get('direction')[()]).reshape(-1, 1).T
        self._type = line.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        return self._location + sample_points * self._direction

    def derivative(self, sample_points, order):
        if order == 1:
            return np.tile(self._direction, (sample_points.shape[0], 1))
        return np.zeros([sample_points.shape[0], self._location.shape[1]])


class Circle(Curve):
    def __init__(self, circle):
        self._location = np.array(circle.get('location')[()]).reshape(-1, 1).T
        self._radius = circle.get('radius')[()]
        self._interval = np.array(circle.get('interval')[()]).reshape(-1, 1)
        self._x_axis = np.array(circle.get('x_axis')[()]).reshape(-1, 1).T
        self._y_axis = np.array(circle.get('y_axis')[()]).reshape(-1, 1).T
        self._type = circle.get('type')[()].decode('utf8')
        if hasattr(circle, 'z_axis'):
            self._z_axis = np.array(circle.get('z_axis')[()]).reshape(-1, 1).T

    def sample(self, sample_points):
        circle_points = self._location + self._radius * (
                np.cos(sample_points) * self._x_axis + np.sin(sample_points) * self._y_axis)
        return circle_points

    def derivative(self, sample_points, order):
        if order % 4 == 0:
            return self.sample(sample_points)
        elif order % 4 == 1:
            return (-self._radius * np.sin(sample_points) * self._x_axis) + (
                    self._radius * np.cos(sample_points) * self._y_axis)
        elif order % 4 == 2:
            return -self.sample(sample_points)
        else:
            return self._radius * (np.sin(sample_points) * self._x_axis - np.cos(sample_points) * self._y_axis)

class Ellipse(Curve):
    def __init__(self, ellipse):
        self._focus1 = np.array(ellipse.get('focus1')[()]).reshape(-1, 1).T
        self._focus2 = np.array(ellipse.get('focus2')[()]).reshape(-1, 1).T
        self._interval = np.array(ellipse.get('interval')[()]).reshape(-1, 1)
        self._maj_radius = ellipse.get('maj_radius')[()]
        self._min_radius = ellipse.get('min_radius')[()]
        self._x_axis = np.array(ellipse.get('x_axis')[()]).reshape(-1, 1).T
        self._y_axis = np.array(ellipse.get('y_axis')[()]).reshape(-1, 1).T
        self._type = ellipse.get('type')[()].decode('utf8')

        if hasattr(ellipse, 'z_axis'):
            self._z_axis = np.array(ellipse.get('z_axis')[()]).reshape(-1, 1).T

        self._center = (self._focus1 + self._focus2) / 2

    def sample(self, sample_points):
        # Check if sample_points are the start and end of the interval, add the midpoint
        if np.array_equal(sample_points, self._interval):
            midpoint = np.array([(self._interval[0] + self._interval[1]) / 2]).reshape(-1, 1)
            sample_points = np.vstack([self._interval[0], midpoint, self._interval[1]])

        ellipse_points = self._center + self._maj_radius * np.cos(sample_points) * self._x_axis + \
                         self._min_radius * np.sin(sample_points) * self._y_axis
        return ellipse_points

    def derivative(self, sample_points, order):
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

class BSplineCurve(Curve):
    def __init__(self, bspline):
        # Attributes initialization for B-spline curve
        self._closed = bspline.get('closed')[()]
        self._degree = bspline.get('degree')[()]
        self._poles = np.array(bspline.get('poles')[()])
        self._knots = np.array(bspline.get('knots')[()]).reshape(-1, 1).T
        self._weights = np.array(bspline.get('weights')[()]).reshape(-1, 1)
        self._interval = np.array(bspline.get('interval')[()]).reshape(-1, 1)
        self._rational = bspline.get('rational')[()]
        self._type = bspline.get('type')[()].decode('utf8')

        # Create BSpline or NURBS curve object
        if not self._rational:
            self._curveObject = BSpline.Curve(normalize_kv=False)
        else:
            self._curveObject = NURBS.Curve(normalize_kv=False)
        self._curveObject.degree = self._degree
        self._curveObject.ctrlpts = self._poles.tolist()
        self._curveObject.knotvector = self._knots.flatten().tolist()
        if self._rational:
            self._curveObject.weights = self._weights.flatten().tolist()

    def sample(self, sample_points):
        return np.array(self._curveObject.evaluate_list(sample_points[:, 0].tolist()))

    def derivative(self, sample_points, order):
        return np.array([
            self._curveObject.derivatives(sample_points[i, 0], order)[-1] for i in range(sample_points.shape[0])
        ])
