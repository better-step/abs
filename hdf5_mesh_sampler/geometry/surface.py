import numpy as np
from geomdl import BSpline


class Surface:
    def sample(self, points):
        raise NotImplementedError("Sample method must be implemented by subclasses")

    def derivative(self, points, order=1):  # commented out = 1
        raise NotImplementedError("Derivative method must be implemented by subclasses")

    def normal(self, points):
        raise NotImplementedError("Normal method must be implemented by subclasses")


class Plane(Surface):
    def __init__(self, plane):
        if isinstance(plane, dict):
            self._location = np.array(plane['location']).reshape(-1, 1).T
            self._coefficients = np.array(plane['coefficients']).reshape(-1, 1).T
            self._trim_domain = np.array(plane['trim_domain'])
            self._x_axis = np.array(plane['x_axis']).reshape(-1, 1).T
            self._y_axis = np.array(plane['y_axis']).reshape(-1, 1).T
            self._z_axis = np.array(plane['z_axis']).reshape(-1, 1).T
            self._type = plane['type']
        else:
            self._location = np.array(plane.get('location')[()]).reshape(-1, 1).T
            self._coefficients = np.array(plane.get('coefficients')[()]).reshape(-1, 1).T
            self._trim_domain = np.array(plane.get('trim_domain')[()])
            self._x_axis = np.array(plane.get('x_axis')[()]).reshape(-1, 1).T
            self._y_axis = np.array(plane.get('y_axis')[()]).reshape(-1, 1).T
            self._z_axis = np.array(plane.get('z_axis')[()]).reshape(-1, 1).T
            self._type = plane.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self._location
        plane_points = self._location.T + sample_points[:, 0] * self._x_axis.T + sample_points[:, 1] * self._y_axis.T
        return plane_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            deriv = np.zeros((sample_points.shape[0], 3, 2))
            deriv[:, :, 0] = self._x_axis
            deriv[:, :, 1] = self._y_axis
            return deriv
        elif order == 2:
            return np.zeros((sample_points.shape[0], 3, 2, 2))
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        # The normal vector is constant for a plane and can be found by crossing x_axis with y_axis
        normal_vector = np.cross(self._x_axis.squeeze(), self._y_axis.squeeze())
        normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
        normals = np.tile(normal_vector_normalized, (sample_points.shape[0], 1))
        return normals

    def shape_type(self):
        return "Plane"


class Cylinder(Surface):
    def __init__(self, cylinder):
        if isinstance(cylinder, dict):
            self._location = np.array(cylinder['location']).reshape(-1, 1).T
            self._radius = cylinder['radius']
            self._coefficients = np.array(cylinder['coefficients']).reshape(-1, 1).T
            self._trim_domain = np.array(cylinder['trim_domain'])
            self._x_axis = np.array(cylinder['x_axis']).reshape(-1, 1).T
            self._y_axis = np.array(cylinder['y_axis']).reshape(-1, 1).T
            self._z_axis = np.array(cylinder['z_axis']).reshape(-1, 1).T
            self._type = cylinder['type']
        else:
            self._location = np.array(cylinder.get('location')[()]).reshape(-1, 1).T
            self._radius = cylinder.get('radius')[()]
            self._coefficients = np.array(cylinder.get('coefficients')[()]).reshape(-1, 1).T
            self._trim_domain = np.array(cylinder.get('trim_domain')[()])
            self._x_axis = np.array(cylinder.get('x_axis')[()]).reshape(-1, 1).T
            self._y_axis = np.array(cylinder.get('y_axis')[()]).reshape(-1, 1).T
            self._z_axis = np.array(cylinder.get('z_axis')[()]).reshape(-1, 1).T
            self._type = cylinder.get('type')[()].decode('utf8')

    def sample(self, sample_points):

        if sample_points.size == 0:
            return self._location

        cylinder_points = self._location.T + self._radius * np.cos(sample_points[:, 0]) * self._x_axis.T + \
                          self._radius * np.sin(sample_points[:, 0]) * self._y_axis.T + sample_points[:, 1] * \
                          self._z_axis.T
        return cylinder_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            dev = np.zeros((sample_points.shape[0], 3, 2))
            dev[:, :, 0] = self._radius * (
                    -np.sin(sample_points[:, 0]) * self._x_axis + np.cos(sample_points[:, 0]) * self._y_axis)
            dev[:, :, 1] = self._z_axis
            return dev
        elif order == 2:
            dev = np.zeros((sample_points.shape[0], 3, 2, 2))
            dev[:, :, 0, 0] = -self._radius * (
                    np.cos(sample_points[:, 0]) * self._x_axis + np.sin(sample_points[:, 0]) * self._y_axis)
            return dev
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        # For a cylinder, compute tangent vectors and their cross product for normals
        derivatives = self.derivative(sample_points, order=1)
        normals = np.cross(derivatives[:, :, 0], derivatives[:, :, 1])
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        return normals

    def shape_type(self):
        return "Cylinder"


class Cone(Surface):
    def __init__(self, cone):
        if isinstance(cone, dict):
            self._location = np.array(cone['location']).reshape(-1, 1).T
            self._radius = cone['radius']
            self._coefficients = np.array(cone['coefficients']).reshape(-1, 1).T
            self._trim_domain = np.array(cone['trim_domain'])
            self._apex = np.array(cone['apex']).reshape(-1, 1).T
            self._angle = cone['angle']
            self._x_axis = np.array(cone['x_axis']).reshape(-1, 1).T
            self._y_axis = np.array(cone['y_axis']).reshape(-1, 1).T
            self._z_axis = np.array(cone['z_axis']).reshape(-1, 1).T
            self._type = cone['type']
        else:
            self._location = np.array(cone.get('location')[()]).reshape(-1, 1).T
            self._radius = cone.get('radius')[()]
            self._coefficients = np.array(cone.get('coefficients')[()]).reshape(-1, 1).T
            self._trim_domain = np.array(cone.get('trim_domain')[()])
            self._apex = np.array(cone.get('apex')[()]).reshape(-1, 1).T
            self._angle = cone.get('angle')[()]
            self._x_axis = np.array(cone.get('x_axis')[()]).reshape(-1, 1).T
            self._y_axis = np.array(cone.get('y_axis')[()]).reshape(-1, 1).T
            self._z_axis = np.array(cone.get('z_axis')[()]).reshape(-1, 1).T
            self._type = cone.get('type')[()].decode('utf8')

    def sample(self, sample_points):

        if sample_points.size == 0:
            return self._apex  #TODO: check this may be to location
        cone_points = self._location.T + (self._radius + sample_points[:, 1] * np.sin(self._angle)) * \
                      (np.cos(sample_points[:, 0]) * self._x_axis.T + np.sin(sample_points[:, 0]) * self._y_axis.T) \
                      + sample_points[:, 1] * np.cos(self._angle) * self._z_axis.T
        return cone_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            dev = np.zeros((sample_points.shape[0], 3, 2))
            dev[:, :, 0] = ((self._radius + sample_points[:, 1] * np.sin(self._angle)) *
                            (-np.sin(sample_points[:, 0]) * self._x_axis + np.cos(sample_points[:, 0]) * self._y_axis))
            dev[:, :, 1] = (np.sin(self._angle) * (
                    np.cos(sample_points[:, 0]) * self._x_axis + np.sin(sample_points[:, 0]) * self._y_axis) +
                            np.cos(self._angle) * self._z_axis)
            return dev
        elif order == 2:
            dev = np.zeros((sample_points.shape[0], 3, 2, 2))
            dev[:, :, 0, 0] = (-(self._radius + sample_points[:, 1] * np.sin(self._angle)) *
                               (np.cos(sample_points[:, 0]) * self._x_axis + np.sin(
                                   sample_points[:, 0]) * self._y_axis))
            dev[:, :, 1, 0] = (np.sin(self._angle) * (
                    -np.sin(sample_points[:, 0]) * self._x_axis + np.cos(sample_points[:, 0]) * self._y_axis))
            dev[:, :, 0, 1] = dev[:, :, 1, 0]
            return dev
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        # For a cylinder, compute tangent vectors and their cross product for normals
        derivatives = self.derivative(sample_points, order=1)
        normals = np.cross(derivatives[:, :, 0], derivatives[:, :, 1])
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        return normals
    def shape_type(self):
        return "Cone"


class Sphere(Surface):
    def __init__(self, sphere):
        if isinstance(sphere, dict):
            self._location = np.array(sphere['location']).reshape(-1, 1).T
            self._radius = sphere['radius']
            self._coefficients = np.array(sphere['coefficients']).reshape(-1, 1).T
            self._trim_domain = np.array(sphere['trim_domain'])
            self._x_axis = np.array(sphere['x_axis']).reshape(-1, 1).T
            self._y_axis = np.array(sphere['y_axis']).reshape(-1, 1).T
            if 'z_axis' in sphere:
                self._z_axis = np.array(sphere['z_axis']).reshape(-1, 1).T
            else:
                self._z_axis = np.zeros((1, 3))
            self._type = sphere['type']
        else:
            self._location = np.array(sphere.get('location')[()]).reshape(-1, 1).T
            self._radius = sphere.get('radius')[()]
            self._coefficients = np.array(sphere.get('coefficients')[()]).reshape(-1, 1).T
            self._trim_domain = np.array(sphere.get('trim_domain')[()])
            self._x_axis = np.array(sphere.get('x_axis')[()]).reshape(-1, 1).T
            self._y_axis = np.array(sphere.get('y_axis')[()]).reshape(-1, 1).T
            if hasattr(sphere, 'z_axis'):
                self._z_axis = np.array(sphere.get('z_axis')[()]).reshape(-1, 1).T
            else:
                self._z_axis = np.zeros((1, 3))
            self._type = sphere.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self._location
        sphere_points = self._location.T + self._radius * np.cos(sample_points[:, 1]) * \
                        (np.cos(sample_points[:, 0]) * self._x_axis.T + np.sin(sample_points[:, 0]) * self._y_axis.T) \
                        + self._radius * np.sin(sample_points[:, 1]) * self._z_axis.T
        return sphere_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            dev = np.zeros((sample_points.shape[0], 3, 2))
            dev[:, :, 0] = (self._radius * np.cos(sample_points[:, 1]) * (
                    -np.sin(sample_points[:, 0]) * self._x_axis + np.cos(sample_points[:, 0]) * self._y_axis))
            dev[:, :, 1] = -self._radius * np.sin(sample_points[:, 1]) * (
                    np.cos(sample_points[:, 0]) * self._x_axis + np.sin(sample_points[:, 0]) * self._y_axis) + \
                           self._radius * np.cos(sample_points[:, 1]) * self._z_axis
            return dev
        elif order == 2:
            dev = np.zeros((sample_points.shape[0], 3, 2, 2))
            dev[:, :, 0, 0] = -self._radius * np.cos(sample_points[:, 1]) * (
                    np.cos(sample_points[:, 0]) * self._x_axis + np.sin(sample_points[:, 0]) * self._y_axis)
            dev[:, :, 0, 1] = -self._radius * np.sin(sample_points[:, 1]) * (
                    -np.sin(sample_points[:, 0]) * self._x_axis + np.cos(sample_points[:, 0]) * self._y_axis)
            dev[:, :, 1, 0] = dev[:, :, 0, 1]
            dev[:, :, 1, 1] = -self._radius * np.cos(sample_points[:, 1]) * (
                    np.cos(sample_points[:, 0]) * self._x_axis + np.sin(sample_points[:, 0]) * self._y_axis) - \
                              self._radius * np.sin(sample_points[:, 1]) * self._z_axis
            return dev
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        # Compute points on the sphere
        sphere_points = self.sample(sample_points)
        # Calculate normals as vectors from sphere center to the points, normalize these vectors
        normals = sphere_points - self._location
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        return normals

    def shape_type(self):
        return "Sphere"



class Torus(Surface):
    def __init__(self, torus):
        if isinstance(torus, dict):
            self._location = np.array(torus['location']).reshape(-1, 1).T
            self._max_radius = torus['max_radius']
            self._min_radius = torus['min_radius']
            self._trim_domain = np.array(torus['trim_domain'])
            self._x_axis = np.array(torus['x_axis']).reshape(-1, 1).T
            self._y_axis = np.array(torus['y_axis']).reshape(-1, 1).T
            self._z_axis = np.array(torus['z_axis']).reshape(-1, 1).T
            self._type = torus['type']
        else:
            self._location = np.array(torus.get('location')[()]).reshape(-1, 1).T
            self._max_radius = torus.get('max_radius')[()]
            self._min_radius = torus.get('min_radius')[()]
            self._trim_domain = np.array(torus.get('trim_domain')[()])
            self._x_axis = np.array(torus.get('x_axis')[()]).reshape(-1, 1).T
            self._y_axis = np.array(torus.get('y_axis')[()]).reshape(-1, 1).T
            self._z_axis = np.array(torus.get('z_axis')[()]).reshape(-1, 1).T
            self._type = torus.get('type')[()].decode('utf8')

    def sample(self, sample_points):
        if sample_points.size == 0:
            return self._location
        torus_points = self._location.T + (self._max_radius + self._min_radius * np.cos(sample_points[:, 1])) * \
                       (np.cos(sample_points[:, 0]) * self._x_axis.T + np.sin(sample_points[:, 0]) * self._y_axis.T) \
                       + self._min_radius * np.sin(sample_points[:, 1]) * self._z_axis.T
        return torus_points.T

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            dev = np.zeros((sample_points.shape[0], 3, 2))
            dev[:, :, 0] = ((self._max_radius + self._min_radius * np.cos(sample_points[:, 1])) *
                            (-np.sin(sample_points[:, 0]) * self._x_axis +
                             np.cos(sample_points[:, 0]) * self._y_axis)).T
            dev[:, :, 1] = ((-self._min_radius * np.sin(sample_points[:, 1])) *
                            (np.cos(sample_points[:, 0]) * self._x_axis +
                             np.sin(sample_points[:, 0]) * self._y_axis) +
                            self._min_radius * np.cos(sample_points[:, 1]) * self._z_axis).T
            return dev
        elif order == 2:
            dev = np.zeros((sample_points.shape[0], 3, 2, 2))
            dev[:, :, 0, 0] = ((self._max_radius + self._min_radius * np.cos(sample_points[:, 1])) *
                               (-np.cos(sample_points[:, 0]) * self._x_axis -
                                np.sin(sample_points[:, 0]) * self._y_axis)).T
            dev[:, :, 0, 1] = ((-self._min_radius * np.sin(sample_points[:, 1])) *
                               (-np.sin(sample_points[:, 0]) * self._x_axis +
                                np.cos(sample_points[:, 0]) * self._y_axis)).T
            dev[:, :, 1, 0] = ((-self._min_radius * np.sin(sample_points[:, 1])) *
                               (-np.sin(sample_points[:, 0]) * self._x_axis +
                                np.cos(sample_points[:, 0]) * self._y_axis)).T
            return dev
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        # Compute derivatives for tangent vectors
        derivatives = self.derivative(sample_points, order=1)
        # Compute normals through cross product and normalize
        normals = np.cross(derivatives[:, :, 0], derivatives[:, :, 1])
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        return normals

    def shape_type(self):
        return "Torus"


class BSplineSurface(Surface):
    def __init__(self, bspline_surface):
        if isinstance(bspline_surface, dict):
            self._continuity = bspline_surface['continuity']
            self._face_domain = np.array(bspline_surface['face_domain']).reshape(-1, 1).T
            self._is_trimmed = bspline_surface['is_trimmed']
            self._poles = np.array(bspline_surface['poles'])
            self._trim_domain = np.array(bspline_surface['trim_domain'])
            self._u_closed = bspline_surface['u_closed']
            self._u_degree = bspline_surface['u_degree']
            self._u_knots = np.array(bspline_surface['u_knots']).reshape(-1, 1).T
            self._u_rational = bspline_surface['u_rational']
            self._v_closed = bspline_surface['v_closed']
            self._v_degree = bspline_surface['v_degree']
            self._v_knots = np.array(bspline_surface['v_knots']).reshape(-1, 1).T
            self._v_rational = bspline_surface['v_rational']
            self._weights = np.array(bspline_surface['weights'])
            self._type = bspline_surface['type']
        else:
            self._continuity = bspline_surface.get('continuity')[()]
            self._face_domain = np.array(bspline_surface.get('face_domain')[()]).reshape(-1, 1).T
            self._is_trimmed = bspline_surface.get('is_trimmed')[()]
            self._poles = np.array(bspline_surface.get('poles')[()])
            self._trim_domain = np.array(bspline_surface.get('trim_domain')[()])
            self._u_closed = bspline_surface.get('u_closed')[()]
            self._u_degree = bspline_surface.get('u_degree')[()]
            self._u_knots = np.array(bspline_surface.get('u_knots')[()]).reshape(-1, 1).T
            self._u_rational = bspline_surface.get('u_rational')[()]
            self._v_closed = bspline_surface.get('v_closed')[()]
            self._v_degree = bspline_surface.get('v_degree')[()]
            self._v_knots = np.array(bspline_surface.get('v_knots')[()]).reshape(-1, 1).T
            self._v_rational = bspline_surface.get('v_rational')[()]
            self._weights = np.column_stack((bspline_surface.get('weights').get('0')[()],
                                            bspline_surface.get('weights').get('1')[()])).reshape(-1, 1)
            self._type = bspline_surface.get('type')[()].decode('utf8')

        self._surface_obj = BSpline.Surface(normalize_kv=False)
        self._surface_obj.degree_u = self._u_degree
        self._surface_obj.degree_v = self._v_degree
        self._surface_obj.ctrlpts_size_u = self._poles.shape[0]
        self._surface_obj.ctrlpts_size_v = self._poles.shape[1]
        self._surface_obj.knotvector_u = self._u_knots.squeeze().tolist()
        self._surface_obj.knotvector_v = self._v_knots.squeeze().tolist()
        self._surface_obj.ctrlpts = self._poles.reshape((-1, 1, 3)).squeeze().tolist()

    def sample(self, sample_points):
        if sample_points.size == 0:
            # Returning the first control point for simplicity
            return self._poles[0, 0]
        uv_pairs = [(sample_points[i, 0], sample_points[i, 1]) for i in range(len(sample_points[:, 0]))]
        return np.array(self._surface_obj.evaluate_list(uv_pairs))

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        return np.array([
            self._surface_obj.derivatives(sample_points[i, 0], sample_points[i, 1], order)[-1] for i in
            range(sample_points.shape[0])
        ])

    def normal(self, sample_points):
        if sample_points.size == 0:
            return np.array([])  # Handle empty input gracefully.

        # Evaluate normals using the geomdl built-in function.
        normals = np.array([self._surface_obj.normal([u, v]) for u, v in sample_points])

        # The returned normals are in the form of tuples (origin, vector components).
        # For consistency and use in graphics or further calculations, you might only need the vector components.
        # Therefore, you may choose to only return these components.
        normal_vectors = normals[:, 1]  # Assuming the second element of each tuple is the vector component.

        return normal_vectors




    def shape_type(self):
        return "BSplineSurface"
