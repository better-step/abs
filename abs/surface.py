import numpy as np
from geomdl import BSpline
from geomdl import operations
from scipy.interpolate import bisplrep, bisplev


class Surface:
    def sample(self, points):
        raise NotImplementedError("Sample method must be implemented by subclasses")

    def derivative(self, points, order=1):
        raise NotImplementedError("Derivative method must be implemented by subclasses")

    def normal(self, points):
        derivatives = self.derivative(points, order=1)
        normals = np.cross(derivatives[:, :, 0], derivatives[:, :, 1])
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        return normals

    def get_area(self):
        if self.area == -1:
            x, w = np.polynomial.legendre.leggauss(4)
            pts = np.array(np.meshgrid(x, x, indexing='ij')).reshape(2, -1).T+1
            pts *= 0.5 * (self.trim_domain[:, 1] - self.trim_domain[:, 0])
            pts += self.trim_domain[:, 0]
            weights = (w * w[:, None]).ravel()

            dd = self.derivative(pts)
            EE = np.sum(dd[:, :, 0] * dd[:, :, 0], axis=1)
            FF = np.sum(dd[:, :, 0] * dd[:, :, 1], axis=1)
            GG = np.sum(dd[:, :, 1] * dd[:, :, 1], axis=1)

            self.area = np.sum(np.sqrt(EE * GG - FF ** 2)*weights)*np.prod(self.trim_domain[:, 1] - self.trim_domain[:, 0]) / 4

        return self.area


class Plane(Surface):
    def __init__(self, plane):
        if isinstance(plane, dict):
            self.location = np.array(plane['location']).reshape(-1, 1).T
            self.coefficients = np.array(plane['coefficients']).reshape(-1, 1).T
            self.trim_domain = np.array(plane['trim_domain'])
            self.x_axis = np.array(plane['x_axis']).reshape(-1, 1).T
            self.y_axis = np.array(plane['y_axis']).reshape(-1, 1).T
            self.z_axis = np.array(plane['z_axis']).reshape(-1, 1).T
            self.area = -1
            self.shape_name = plane['type']
        else:
            self.location = np.array(plane.get('location')[()]).reshape(-1, 1).T
            self.coefficients = np.array(plane.get('coefficients')[()]).reshape(-1, 1).T
            self.trim_domain = np.array(plane.get('trim_domain')[()])
            self.x_axis = np.array(plane.get('x_axis')[()]).reshape(-1, 1).T
            self.y_axis = np.array(plane.get('y_axis')[()]).reshape(-1, 1).T
            self.z_axis = np.array(plane.get('z_axis')[()]).reshape(-1, 1).T
            self.area = -1
            self.shape_name = plane.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self.location
        plane_points = self.location.T + sample_points[:, 0] * self.x_axis.T + sample_points[:, 1] * self.y_axis.T
        return plane_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            deriv = np.zeros((sample_points.shape[0], 3, 2))
            deriv[:, :, 0] = self.x_axis
            deriv[:, :, 1] = self.y_axis
            return deriv
        elif order == 2:
            return np.zeros((sample_points.shape[0], 3, 2, 2))
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        # The normal vector is constant for a plane and can be found by crossing x_axis with y_axis
        normal_vector = np.cross(self.x_axis.squeeze(), self.y_axis.squeeze())
        normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
        normals = np.tile(normal_vector_normalized, (sample_points.shape[0], 1))
        return normals


class Cylinder(Surface):
    def __init__(self, cylinder):
        if isinstance(cylinder, dict):
            self.location = np.array(cylinder['location']).reshape(-1, 1).T
            self.radius = float(cylinder['radius'])
            self.coefficients = np.array(cylinder['coefficients']).reshape(-1, 1).T
            self.trim_domain = np.array(cylinder['trim_domain'])
            self.x_axis = np.array(cylinder['x_axis']).reshape(-1, 1).T
            self.y_axis = np.array(cylinder['y_axis']).reshape(-1, 1).T
            self.z_axis = np.array(cylinder['z_axis']).reshape(-1, 1).T
            self.area = -1
            self.shape_name = cylinder['type']
        else:
            self.location = np.array(cylinder.get('location')[()]).reshape(-1, 1).T
            self.radius = float(cylinder.get('radius')[()])
            self.coefficients = np.array(cylinder.get('coefficients')[()]).reshape(-1, 1).T
            self.trim_domain = np.array(cylinder.get('trim_domain')[()])
            self.x_axis = np.array(cylinder.get('x_axis')[()]).reshape(-1, 1).T
            self.y_axis = np.array(cylinder.get('y_axis')[()]).reshape(-1, 1).T
            self.z_axis = np.array(cylinder.get('z_axis')[()]).reshape(-1, 1).T
            self.area = -1
            self.shape_name = cylinder.get('type')[()].decode('utf8')

    def sample(self, sample_points):

        if sample_points.size == 0:
            return self.location

        cylinder_points = self.location.T + self.radius * np.cos(sample_points[:, 0]) * self.x_axis.T + \
                          self.radius * np.sin(sample_points[:, 0]) * self.y_axis.T + sample_points[:, 1] * \
                          self.z_axis.T
        return cylinder_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            dev = np.zeros((sample_points.shape[0], 3, 2))
            dev[:, :, 0] = self.radius * \
                           (-np.sin(sample_points[:, 0]) * self.x_axis.T +
                            np.cos(sample_points[:, 0]) * self.y_axis.T).T
            dev[:, :, 1] = self.z_axis
            return dev
        elif order == 2:
            dev = np.zeros((sample_points.shape[0], 3, 2, 2))
            dev[:, :, 0, 0] = (-self.radius * (np.cos(sample_points[:, 0]) * self.x_axis.T +
                                                np.sin(sample_points[:, 0]) * self.y_axis.T)).T
            return dev
        else:
            raise ValueError("Order must be 0, 1, or 2")


class Cone(Surface):
    def __init__(self, cone):
        if isinstance(cone, dict):
            self.location = np.array(cone['location']).reshape(-1, 1).T
            self.radius = float(cone['radius'])
            self.coefficients = np.array(cone['coefficients']).reshape(-1, 1).T
            self.trim_domain = np.array(cone['trim_domain'])
            self.apex = np.array(cone['apex']).reshape(-1, 1).T
            self.angle = float(cone['angle'])
            self.x_axis = np.array(cone['x_axis']).reshape(-1, 1).T
            self.y_axis = np.array(cone['y_axis']).reshape(-1, 1).T
            self.z_axis = np.array(cone['z_axis']).reshape(-1, 1).T
            self.area = -1
            self.shape_name = cone['type']
        else:
            self.location = np.array(cone.get('location')[()]).reshape(-1, 1).T
            self.radius = float(cone.get('radius')[()])
            self.coefficients = np.array(cone.get('coefficients')[()]).reshape(-1, 1).T
            self.trim_domain = np.array(cone.get('trim_domain')[()])
            self.apex = np.array(cone.get('apex')[()]).reshape(-1, 1).T
            self.angle = float(cone.get('angle')[()])
            self.x_axis = np.array(cone.get('x_axis')[()]).reshape(-1, 1).T
            self.y_axis = np.array(cone.get('y_axis')[()]).reshape(-1, 1).T
            self.z_axis = np.array(cone.get('z_axis')[()]).reshape(-1, 1).T
            self.area = -1
            self.shape_name = cone.get('type')[()].decode('utf8')

    def sample(self, sample_points):

        if sample_points.size == 0:
            return self.apex
        cone_points = self.location.T + (self.radius + sample_points[:, 1] * np.sin(self.angle)) * \
                      (np.cos(sample_points[:, 0]) * self.x_axis.T + np.sin(sample_points[:, 0]) * self.y_axis.T) \
                      + sample_points[:, 1] * np.cos(self.angle) * self.z_axis.T
        return cone_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            dev = np.zeros((sample_points.shape[0], 3, 2))
            dev[:, :, 0] = ((self.radius + sample_points[:, 1] * np.sin(self.angle)) *
                            (-np.sin(sample_points[:, 0]) * self.x_axis.T +
                             np.cos(sample_points[:, 0]) * self.y_axis.T)).T
            dev[:, :, 1] = (np.sin(self.angle) *
                            (np.cos(sample_points[:, 0]) * self.x_axis.T + np.sin(sample_points[:, 0]) *
                             self.y_axis.T) + np.cos(self.angle) * self.z_axis.T).T
            return dev
        elif order == 2:
            dev = np.zeros((sample_points.shape[0], 3, 2, 2))
            dev[:, :, 0, 0] = ((self.radius + sample_points[:, 1] * np.sin(self.angle)) *
                               (-np.cos(sample_points[:, 0]) * self.x_axis.T -
                                np.sin(sample_points[:, 0]) * self.y_axis.T)).T
            dev[:, :, 0, 1] = (np.sin(self.angle) * (-np.sin(sample_points[:, 0]) *
                                                      self.x_axis.T + np.cos(sample_points[:, 0]) * self.y_axis.T).T)
            dev[:, :, 1, 0] = (np.sin(self.angle) * (-np.sin(sample_points[:, 0]) *
                                                      self.x_axis.T + np.cos(sample_points[:, 0]) * self.y_axis.T).T)
            return dev
        else:
            raise ValueError("Order must be 0, 1, or 2")


class Sphere(Surface):
    def __init__(self, sphere):
        if isinstance(sphere, dict):
            self.location = np.array(sphere['location']).reshape(-1, 1).T
            self.radius = float(sphere['radius'])
            self.coefficients = np.array(sphere['coefficients']).reshape(-1, 1).T
            self.trim_domain = np.array(sphere['trim_domain'])
            self.x_axis = np.array(sphere['x_axis']).reshape(-1, 1).T
            self.y_axis = np.array(sphere['y_axis']).reshape(-1, 1).T
            if 'z_axis' in sphere:
                self.z_axis = np.array(sphere['z_axis']).reshape(-1, 1).T
            else:
                self.z_axis = np.zeros((1, 3))
            self.area = -1
            self.shape_name = sphere['type']
        else:
            self.location = np.array(sphere.get('location')[()]).reshape(-1, 1).T
            self.radius = float(sphere.get('radius')[()])
            self.coefficients = np.array(sphere.get('coefficients')[()]).reshape(-1, 1).T
            self.trim_domain = np.array(sphere.get('trim_domain')[()])
            self.x_axis = np.array(sphere.get('x_axis')[()]).reshape(-1, 1).T
            self.y_axis = np.array(sphere.get('y_axis')[()]).reshape(-1, 1).T
            if 'z_axis' in sphere:
                self.z_axis = np.array(sphere.get('z_axis')[()]).reshape(-1, 1).T
            else:
                self.z_axis = np.cross(self.x_axis, self.y_axis)
            self.area = -1
            self.shape_name = sphere.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self.location
        sphere_points = self.location.T + self.radius * np.cos(sample_points[:, 1]) * \
                        (np.cos(sample_points[:, 0]) * self.x_axis.T + np.sin(sample_points[:, 0]) * self.y_axis.T) \
                        + self.radius * np.sin(sample_points[:, 1]) * self.z_axis.T
        return sphere_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            dev = np.zeros((sample_points.shape[0], 3, 2))
            dev[:, :, 0] = (self.radius * np.cos(sample_points[:, 1]))[:, np.newaxis] * \
                           (-np.sin(sample_points[:, 0]) * self.x_axis.T +
                            np.cos(sample_points[:, 0]) * self.y_axis.T).T
            dev[:, :, 1] = -self.radius * np.sin(sample_points[:, 1])[:, np.newaxis] * \
                           (np.cos(sample_points[:, 0]) * self.x_axis.T +
                            np.sin(sample_points[:, 0]) * self.y_axis.T).T + \
                           (self.radius * np.cos(sample_points[:, 1]) * self.z_axis.T).T
            return dev
        elif order == 2:
            dev = np.zeros((sample_points.shape[0], 3, 2, 2))
            dev[:, :, 0, 0] = (self.radius * np.cos(sample_points[:, 1]) *
                               (-np.cos(sample_points[:, 0]) *
                                self.x_axis.T - np.sin(sample_points[:, 0]) * self.y_axis.T)).T
            dev[:, :, 0, 1] = (-self.radius * np.sin(sample_points[:, 1])
                               * (-np.sin(sample_points[:, 0]) *
                                  self.x_axis.T + np.cos(sample_points[:, 0]) * self.y_axis.T)).T
            dev[:, :, 1, 0] = (-self.radius * np.sin(sample_points[:, 1]) *
                               (-np.sin(sample_points[:, 0]) * self.x_axis.T +
                                np.cos(sample_points[:, 0]) * self.y_axis.T)).T
            dev[:, :, 1, 1] = (-self.radius * np.cos(sample_points[:, 1]) *
                               (np.cos(sample_points[:, 0]) * self.x_axis.T +
                                np.sin(sample_points[:, 0]) * self.y_axis.T) -
                               self.radius * np.sin(sample_points[:, 1]) * self.z_axis.T).T
            return dev
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        sphere_points = self.sample(sample_points)
        normals = sphere_points - self.location
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        return normals


class Torus(Surface):
    def __init__(self, torus):
        if isinstance(torus, dict):
            self.location = np.array(torus['location']).reshape(-1, 1).T
            self.max_radius = float(torus['max_radius'])
            self.min_radius = float(torus['min_radius'])
            self.trim_domain = np.array(torus['trim_domain'])
            self.x_axis = np.array(torus['x_axis']).reshape(-1, 1).T
            self.y_axis = np.array(torus['y_axis']).reshape(-1, 1).T
            self.z_axis = np.array(torus['z_axis']).reshape(-1, 1).T
            self.area = -1
            self.shape_name = torus['type']
        else:
            self.location = np.array(torus.get('location')[()]).reshape(-1, 1).T
            self.max_radius = float(torus.get('max_radius')[()])
            self.min_radius = float(torus.get('min_radius')[()])
            self.trim_domain = np.array(torus.get('trim_domain')[()])
            self.x_axis = np.array(torus.get('x_axis')[()]).reshape(-1, 1).T
            self.y_axis = np.array(torus.get('y_axis')[()]).reshape(-1, 1).T
            self.z_axis = np.array(torus.get('z_axis')[()]).reshape(-1, 1).T
            self.area = -1
            self.shape_name = torus.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self.location
        torus_points = self.location.T + (self.max_radius + self.min_radius * np.cos(sample_points[:, 1])) * \
                       (np.cos(sample_points[:, 0]) * self.x_axis.T + np.sin(sample_points[:, 0]) * self.y_axis.T) \
                       + self.min_radius * np.sin(sample_points[:, 1]) * self.z_axis.T
        return torus_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            dev = np.zeros((sample_points.shape[0], 3, 2))

            dev[:, :, 0] = ((self.max_radius + self.min_radius * np.cos(sample_points[:, 1])) *
                            (-np.sin(sample_points[:, 0]) * self.x_axis.T +
                             np.cos(sample_points[:, 0]) * self.y_axis.T)).T

            dev[:, :, 1] = (((-self.min_radius * np.sin(sample_points[:, 1])) *
                             (np.cos(sample_points[:, 0]) * self.x_axis.T +
                              np.sin(sample_points[:, 0]) * self.y_axis.T)) +
                            (self.min_radius * np.cos(sample_points[:, 1])) * self.z_axis.T).T
            return dev
        elif order == 2:
            dev = np.zeros((sample_points.shape[0], 3, 2, 2))
            dev[:, :, 0, 0] = ((self.max_radius + self.min_radius * np.cos(sample_points[:, 1])) *
                               (-np.cos(sample_points[:, 0]) * self.x_axis.T -
                                np.sin(sample_points[:, 0]) * self.y_axis.T)).T

            dev[:, :, 0, 1] = (-self.min_radius * np.sin(sample_points[:, 1]) *
                               (-np.sin(sample_points[:, 0]) * self.x_axis.T +
                                np.cos(sample_points[:, 0]) * self.y_axis.T)).T

            dev[:, :, 1, 0] = (-self.min_radius * np.sin(sample_points[:, 1]) *
                               (-np.sin(sample_points[:, 0]) * self.x_axis.T +
                                np.cos(sample_points[:, 0]) * self.y_axis.T)).T

            dev[:, :, 1, 1] = ((-self.min_radius * np.cos(sample_points[:, 1]) *
                                (np.cos(sample_points[:, 0]) * self.x_axis.T + np.sin(sample_points[:, 0])
                                 * self.y_axis.T)) - (self.min_radius *
                                                       np.sin(sample_points[:, 1]) * self.z_axis.T)).T
            return dev
        else:
            raise ValueError("Order must be 0, 1, or 2")


class BSplineSurface(Surface):
    def __init__(self, bspline_surface):
        if isinstance(bspline_surface, dict):
            self.continuity = int(bspline_surface['continuity'])
            self.face_domain = np.array(bspline_surface['face_domain']).reshape(-1, 1).T
            self.is_trimmed = bool(bspline_surface['is_trimmed'])
            self.poles = np.array(bspline_surface['poles'])
            self.trim_domain = np.array(bspline_surface['trim_domain'])
            if len(self.trim_domain.shape) == 1:
                self.trim_domain = np.reshape(self.trim_domain, [2, 2])
            self.u_closed = bool(bspline_surface['u_closed'])
            self.u_degree = int(bspline_surface['u_degree'])
            self.u_knots = np.array(bspline_surface['u_knots']).reshape(-1, 1).T
            self.u_rational = bool(bspline_surface['u_rational'])
            self.v_closed = bool(bspline_surface['v_closed'])
            self.v_degree = int(bspline_surface['v_degree'])
            self.v_knots = np.array(bspline_surface['v_knots']).reshape(-1, 1).T
            self.v_rational = bool(bspline_surface['v_rational'])
            self.weights = np.array(bspline_surface['weights'])
            self.area = -1
            self.shape_name = bspline_surface['type']
        else:
            self.continuity = int(bspline_surface.get('continuity')[()])
            self.face_domain = np.array(bspline_surface.get('face_domain')[()]).reshape(-1, 1).T
            self.is_trimmed = bool(bspline_surface.get('is_trimmed')[()])
            self.poles = np.array(bspline_surface.get('poles')[()])
            self.trim_domain = np.array(bspline_surface.get('trim_domain')[()])
            if len(self.trim_domain.shape) == 1:
                self.trim_domain = np.reshape(self.trim_domain, [2, 2])
            self.u_closed = bool(bspline_surface.get('u_closed')[()])
            self.u_degree = int(bspline_surface.get('u_degree')[()])
            self.u_knots = np.array(bspline_surface.get('u_knots')[()]).reshape(-1, 1).T
            self.u_rational = bool(bspline_surface.get('u_rational')[()])
            self.v_closed = bool(bspline_surface.get('v_closed')[()])
            self.v_degree = int(bspline_surface.get('v_degree')[()])
            self.v_knots = np.array(bspline_surface.get('v_knots')[()]).reshape(-1, 1).T
            self.v_rational = bool(bspline_surface.get('v_rational')[()])
            self.weights = np.column_stack((bspline_surface.get('weights').get('0')[()],
                                             bspline_surface.get('weights').get('1')[()])).reshape(-1, 1)
            self.area = -1
            self.shape_name = bspline_surface.get('type')[()].decode('utf8')

        self.surface_obj = BSpline.Surface(normalize_kv=False)
        self.surface_obj.degree_u = self.u_degree
        self.surface_obj.degree_v = self.v_degree
        self.surface_obj.ctrlpts_size_u = self.poles.shape[0]
        self.surface_obj.ctrlpts_size_v = self.poles.shape[1]
        self.surface_obj.knotvector_u = self.u_knots.squeeze().tolist()
        self.surface_obj.knotvector_v = self.v_knots.squeeze().tolist()
        self.surface_obj.ctrlpts = self.poles.reshape((-1, 1, 3)).squeeze().tolist()

    def sample(self, sample_points):
        if sample_points.size == 0:
            # Returning the first control point for simplicity
            return self.poles[0, 0]
        uv_pairs = [(sample_points[i, 0], sample_points[i, 1]) for i in range(len(sample_points[:, 0]))]
        return np.array(self.surface_obj.evaluate_list(uv_pairs))

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            res = np.zeros((sample_points.shape[0], 3, 2))
            for i in range(sample_points.shape[0]):
                d = self.surface_obj.derivatives(sample_points[i, 0], sample_points[i, 1], order)
                res[i, :, 0] = d[1][0]
                res[i, :, 1] = d[0][1]

            return res
        elif order == 2:
            res = np.zeros((sample_points.shape[0], 3, 2, 2))
            for i in range(sample_points.shape[0]):
                d = self.surface_obj.derivatives(sample_points[i, 0], sample_points[i, 1], order)
                res[i, :, 0, 0] = d[2][0]
                res[i, :, 1, 0] = d[1][1]
                res[i, :, 0, 1] = d[1][1]
                res[i, :, 1, 1] = d[0][2]

            return res
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        if sample_points.size == 0:
            return np.array([])  # Handle empty input gracefully.

        # Evaluate normals using the geomdl built-in function.
        normals = operations.normal(self.surface_obj, sample_points)

        # The returned normals are in the form of tuples (origin, vector components).
        # For consistency and use in graphics or further calculations, you might only need the vector components.
        # Therefore, you may choose to only return these components.
        normal_vectors = np.array([n[-1] for n in normals])

        return normal_vectors
