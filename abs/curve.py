import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BSpline

class Curve:
    def sample(self, points):
        raise NotImplementedError("Sample method must be implemented by subclasses")

    def get_length(self):
        if self.length == -1:
            pts, weights = np.polynomial.legendre.leggauss(4)
            pts += 1
            pts *= 0.5 * (self.interval[0, 1] - self.interval[0, 0])
            pts += self.interval[0, 0]

            dd = self.derivative(pts[:, None], order=1)
            circumference = np.sum(np.linalg.norm(dd, axis=1)*weights)*(self.interval[0, 1] - self.interval[0, 0]) / 2

            self.length = circumference
        return self.length

    def derivative(self, points, order=1):
        raise NotImplementedError("Derivative method must be implemented by subclasses")

    def normal(self, points):
        raise NotImplementedError("Normal method must be implemented by subclasses")


class Line(Curve):
    def __init__(self, line):
        if isinstance(line, dict):
            self.location = np.array(line['location']).reshape(-1, 1).T
            self.interval = np.array(line['interval']).reshape(-1, 1).T
            self.direction = np.array(line['direction']).reshape(-1, 1).T
            if self.direction.shape[1] == 3:
                self.transform = np.array(line['transform'])
            self.length = -1
            self.shape_name = line['type']
        else:
            self.location = np.array(line.get('location')[()]).reshape(-1, 1).T
            self.interval = np.array(line.get('interval')[()]).reshape(-1, 1).T
            self.direction = np.array(line.get('direction')[()]).reshape(-1, 1).T
            if self.direction.shape[1] == 3:
                self.transform = np.array(line.get('transform')[()])
            self.length = -1
            self.shape_name = line.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self.location
        return self.location + sample_points * self.direction

    def get_length(self):
        if self.length == -1:
            self.length = np.linalg.norm((self.interval[0, 1] - self.interval[0, 0]) * self.direction)
        return self.length

    def derivative(self, sample_points, order=1):
        if order == 1:
            return np.tile(self.direction, (sample_points.shape[0], 1))
        return np.zeros([sample_points.shape[0], self.location.shape[1]])

    def normal(self, sample_points):
        normal_vector = np.array([-self.direction[0, 1], self.direction[0, 0]])
        return np.tile(normal_vector, (sample_points.shape[0], 1))


class Circle(Curve):
    def __init__(self, circle):
        if isinstance(circle, dict):
            self.location = np.array(circle['location']).reshape(-1, 1).T
            self.radius = float(circle['radius'])
            self.interval = np.array(circle['interval']).reshape(-1, 1).T
            self.x_axis = np.array(circle['x_axis']).reshape(-1, 1).T
            self.y_axis = np.array(circle['y_axis']).reshape(-1, 1).T
            self.length = -1
            self.shape_name = circle['type']
            if 'z_axis' in circle:
                self.z_axis = np.array(circle['z_axis']).reshape(-1, 1).T
                self.transform = np.array(circle['transform'])
        else:
            self.location = np.array(circle.get('location')[()]).reshape(-1, 1).T
            self.radius = float(circle.get('radius')[()])
            self.interval = np.array(circle.get('interval')[()]).reshape(-1, 1).T
            self.x_axis = np.array(circle.get('x_axis')[()]).reshape(-1, 1).T
            self.y_axis = np.array(circle.get('y_axis')[()]).reshape(-1, 1).T
            self.length = -1
            self.shape_name = circle.get('type')[()].decode('utf8')
            if 'z_axis' in circle:
                self.z_axis = np.array(circle.get('z_axis')[()]).reshape(-1, 1).T
                self.transform = np.array(circle.get('transform')[()])

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self.location
        circle_points = self.location + self.radius * (
            np.cos(sample_points) * self.x_axis + np.sin(sample_points) * self.y_axis)
        return circle_points

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order % 4 == 0 and order != 0:
            return self.radius * (np.cos(sample_points) * self.x_axis + np.sin(sample_points) * self.y_axis)
        elif order % 4 == 1:
            return (-self.radius * np.sin(sample_points) * self.x_axis) + (
                self.radius * np.cos(sample_points) * self.y_axis)
        elif order % 4 == 2:
            return -(self.radius * (np.cos(sample_points) * self.x_axis + np.sin(sample_points) * self.y_axis))
        elif order % 4 == 3:
            return self.radius * (np.sin(sample_points) * self.x_axis - np.cos(sample_points) * self.y_axis)

    def normal(self, sample_points):

        # rotation_matrix = np.array([[0, 1], [-1, 0]])
        # normal_vector = self.derivative(sample_points, order=1) @ rotation_matrix.T
        # normal_vector /= np.linalg.norm(normal_vector, axis=1, keepdims=True)


        first_deriv = self.derivative(sample_points, order=1)

        # calculating the speed
        v = np.linalg.norm(first_deriv, axis=1)

        # calculating the tangent
        tangent = first_deriv / v[:, None]

        t_prime = np.gradient(tangent, axis=0)

        # calculating the normal
        normal = t_prime - np.einsum('ij,ij->i', t_prime, tangent)[:, None] * tangent
        normal_vector = normal / np.linalg.norm(normal, axis=1)[:, None]

        # calculating the binormal (to make sure they are orthogonal)
        binormal = np.cross(tangent, normal)


        return normal_vector


class Ellipse(Curve):
    def __init__(self, ellipse):
        if isinstance(ellipse, dict):
            self.focus1 = np.array(ellipse['focus1']).reshape(-1, 1).T
            self.focus2 = np.array(ellipse['focus2']).reshape(-1, 1).T
            self.interval = np.array(ellipse['interval']).reshape(-1, 1).T
            self.maj_radius = float(ellipse['maj_radius'])
            self.min_radius = float(ellipse['min_radius'])
            self.x_axis = np.array(ellipse['x_axis']).reshape(-1, 1).T
            self.y_axis = np.array(ellipse['y_axis']).reshape(-1, 1).T
            self.length = -1
            self.shape_name = ellipse['type']

            if 'z_axis' in ellipse:
                self.z_axis = np.array(ellipse['z_axis']).reshape(-1, 1).T
                self.transform = np.array(ellipse['transform'])
        else:
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
                self.transform = np.array(ellipse.get('transform')[()])

        self.center = (self.focus1 + self.focus2) / 2

    def sample(self, sample_points):
        ellipse_points = self.center + self.maj_radius * np.cos(sample_points) * self.x_axis + \
                         self.min_radius * np.sin(sample_points) * self.y_axis
        return ellipse_points

    def derivative(self, sample_points, order=1):
        if order % 4 == 0:
            if order == 0:
                return self.sample(sample_points)
            return self.maj_radius * np.cos(sample_points) * self.x_axis + \
                self.min_radius * np.sin(sample_points) * self.y_axis
        elif order % 4 == 1:
            return -self.maj_radius * np.sin(sample_points) * self.x_axis + \
                self.min_radius * np.cos(sample_points) * self.y_axis
        elif order % 4 == 2:
            return -self.maj_radius * np.cos(sample_points) * self.x_axis - \
                self.min_radius * np.sin(sample_points) * self.y_axis
        return self.maj_radius * np.sin(sample_points) * self.x_axis - \
            self.min_radius * np.cos(sample_points) * self.y_axis

    def normal(self, sample_points):

        if hasattr(self, 'z_axis'):

            first_deriv = self.derivative(sample_points, order=1)
            v = np.linalg.norm(first_deriv, axis=1)
            tangent = first_deriv / v[:, None]
            t_prime = np.gradient(tangent, axis=0)
            normal = t_prime - np.einsum('ij,ij->i', t_prime, tangent)[:, None] * tangent
            normal_vector = normal / np.linalg.norm(normal, axis=1)[:, None]
            # binormal = np.cross(tangent, normal)

        else:

            rotation_matrix = np.array([[0, 1], [-1, 0]])
            normal_vector = self.derivative(sample_points, order=1) @ rotation_matrix.T
            normal_vector /= np.linalg.norm(normal_vector, axis=1, keepdims=True)

        return normal_vector


class BSplineCurve(Curve):
    def __init__(self, bspline):

        self.length = -1
        if isinstance(bspline, dict):
            self.closed = bool(bspline['closed'])
            self.degree = int(bspline['degree'])
            self.continuity = int(bspline['continuity'])
            self.poles = np.array(bspline['poles'])
            self.knots = np.array(bspline['knots']).reshape(-1, 1).T
            self.weights = np.array(bspline['weights']).reshape(-1, 1)
            self.interval = np.array(bspline['interval']).reshape(-1, 1).T
            self.rational = bool(bspline['rational'])
            self.periodic = bool(bspline['periodic'])
            self.shape_name = bspline['type']
            if self.poles.shape[1] == 3:
                self.transform = np.array(bspline['transform'])
        else:
            self.closed = bool(bspline.get('closed')[()])
            self.degree = int(bspline.get('degree')[()])
            self.continuity = int(bspline.get('continuity')[()])
            self.poles = np.array(bspline.get('poles')[()])
            self.knots = np.array(bspline.get('knots')[()]).reshape(-1, 1).T
            self.weights = np.array(bspline.get('weights')[()]).reshape(-1, 1)
            self.interval = np.array(bspline.get('interval')[()]).reshape(-1, 1).T
            self.rational = bool(bspline.get('rational')[()])
            self.periodic = bool(bspline.get('periodic')[()])
            self.shape_name = bspline.get('type')[()].decode('utf8')
            if self.poles.shape[1] == 3:
                self.transform = np.array(bspline.get('transform')[()])

        # Create BSpline or NURBS curve object


        # if self.rational:
        #     self.curveObject = NURBS.Curve(normalize_kv=False)
        # else:
        #     self.curveObject = BSpline.Curve(normalize_kv=False)
        #
        # self.curveObject.degree = self.degree
        # self.curveObject.ctrlpts = self.poles.tolist()
        # self.curveObject.knotvector = self.knots.flatten().tolist()
        # if self.rational:
        #     self.curveObject.weights = self.weights.flatten().tolist()

        ctrl_pnts = self.poles
        if self.rational:
            ctrl_pnts = self.poles * self.weights

        self.bspline = BSpline(self.knots.T[:,0], ctrl_pnts, self.degree)


    def sample(self, sample_points):
        # if sample_points.size == 1:
        #     samples = np.array(self.curveObject.evaluate_single(sample_points[0]))
        #     samples = samples.reshape(-1, 1)
        #     return samples.T
        #
        # # Evaluate the curve at the given sample points
        # return np.array(self.curveObject.evaluate_list(sample_points[:, 0].tolist()))

        return np.squeeze(self.bspline(sample_points))

    def derivative(self, sample_points, order=1):

        if order == 0:
            return self.sample(sample_points)

        elif self.degree < order:
            return np.zeros([sample_points.shape[0], self.poles.shape[1]])
        else:
            # res = np.zeros([sample_points.shape[0], self.poles.shape[1]])
            # for i in range(sample_points.shape[0]):
            #     d = self.curveObject.derivatives(sample_points[i, 0], order)
            #     res[i, :] = d[-1]
            # return res
            b_spline_derivative = self.bspline.derivative(order)
            return np.squeeze(b_spline_derivative(sample_points))

    def normal(self, sample_points):
        if sample_points.size == 0:
            return np.array([])

        eps = 1e-8
        first_deriv = self.derivative(sample_points, order=1)
        v = np.linalg.norm(first_deriv, axis=1)
        tangent = first_deriv / v[:, None]
        t_prime = np.gradient(tangent, axis=0)
        proj = np.einsum('ij,ij->i', t_prime, tangent)[:, None] * tangent
        normal = t_prime - proj

        normal_norm = np.linalg.norm(normal, axis=1)
        normal_vector = normal / normal_norm[:, None]
        #binormal = np.cross(tangent, normal)

        return normal_vector

class Other(Curve):
    def __init__(self, other):
        self.shape_name = other.get('type')[()].decode('utf8')
        self.interval = np.array(other.get('interval')[()]).reshape(-1, 1).T

    def sample(self, sample_points):
        return np.array([])

    def derivative(self, sample_points, order=1):
        return np.array([])

    def normal(self, sample_points):
        return np.array([])
