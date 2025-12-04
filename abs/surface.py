"""
Surface classes for various surface types (Plane, Cylinder, Cone, Sphere, Torus, etc.).
Provide methods to sample points on surfaces and compute derivatives and normals.
"""

import numpy as np
from geomdl import operations
from .curve import create_curve
from scipy.stats import skew


def create_surface(surface_data, compute_index=True):
    if compute_index:
        index = int(surface_data.name.split("/")[-1])
    else:
        index = None

    if isinstance(surface_data, dict):
        surface_type = surface_data.get('type')
    else:
        surface_type = surface_data.get('type')[()].decode('utf-8')
    surface_map = {
        'Plane': Plane,
        'Cylinder': Cylinder,
        'Cone': Cone,
        'Sphere': Sphere,
        'Torus': Torus,
        'BSpline': BSplineSurface,
        'Extrusion': Extrusion,
        'Revolution': Revolution,
        'Offset': Offset
    }
    surface_class = surface_map.get(surface_type)
    if surface_class:
        return index, surface_class(surface_data)
    else:
        # Unsupported surface type
        return index, None

class Surface:
    """Base class for all surface types."""
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
        if getattr(self, "area", None) is not None and self.area != -1:
            return self.area
        # Approximate area via 4x4 Gauss-Legendre quadrature

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

    def __eq__(self, other):
        if not isinstance(other, Surface):
            return NotImplemented
        if type(self) is not type(other):
            print(f"[Surface ==] Type mismatch: {type(self).__name__} vs {type(other).__name__}")
            return False

        rtol = 1e-8
        atol = 1e-10

        def _cmp(v1, v2, name):
            if isinstance(v1, (np.ndarray, list, tuple)) or isinstance(v2, (np.ndarray, list, tuple)):
                a1 = np.asarray(v1)
                a2 = np.asarray(v2)
                if a1.shape != a2.shape:
                    print(f"[Surface ==] Attribute '{name}' shape mismatch: {a1.shape} vs {a2.shape}")
                    return False
                if np.issubdtype(a1.dtype, np.number) and np.issubdtype(a2.dtype, np.number):
                    if not np.allclose(a1, a2, rtol=rtol, atol=atol):
                        print(f"[Surface ==] Attribute '{name}' numeric values differ")
                        return False
                    return True
                if not np.array_equal(a1, a2):
                    print(f"[Surface ==] Attribute '{name}' array values differ")
                    return False
                return True

            if isinstance(v1, (int, float, bool, np.generic, str)) and isinstance(v2,
                                                                                  (int, float, bool, np.generic, str)):
                if v1 != v2:
                    print(f"[Surface ==] Attribute '{name}' scalar mismatch: {v1!r} vs {v2!r}")
                    return False
                return True

            if v1 != v2:
                print(f"[Surface ==] Attribute '{name}' mismatch (fallback compare): {v1!r} vs {v2!r}")
                return False
            return True

        ignore = {"area", "surface_obj"}
        attrs1 = {k: v for k, v in self.__dict__.items()
                  if not k.startswith("_") and k not in ignore}
        attrs2 = {k: v for k, v in other.__dict__.items()
                  if not k.startswith("_") and k not in ignore}

        if attrs1.keys() != attrs2.keys():
            only1 = attrs1.keys() - attrs2.keys()
            only2 = attrs2.keys() - attrs1.keys()
            print(f"[Surface ==] Attribute key mismatch. "
                  f"Only in self: {sorted(only1)}, only in other: {sorted(only2)}")
            return False

        for name in attrs1.keys():
            if not _cmp(attrs1[name], attrs2[name], name):
                return False

        return True


class Plane(Surface):
    """Plane surface."""
    def __init__(self, plane, location=None, coefficients=None, trim_domain=None, x_axis=None, y_axis=None, z_axis=None, transform=None):
        self.area = -1

        if plane is None:
            self.location = location.reshape(-1, 1).T
            self.coefficients = coefficients.reshape(-1, 1).T
            self.trim_domain = trim_domain
            self.x_axis = x_axis.reshape(-1, 1).T
            self.y_axis = y_axis.reshape(-1, 1).T
            self.z_axis = z_axis.reshape(-1, 1).T
            self.transform = transform
            self.shape_name = 'Plane'
        else:
            self.location = np.array(plane.get('location')[()]).reshape(-1, 1).T
            self.coefficients = np.array(plane.get('coefficients')[()]).reshape(-1, 1).T
            self.trim_domain = np.array(plane.get('trim_domain')[()])
            self.x_axis = np.array(plane.get('x_axis')[()]).reshape(-1, 1).T
            self.y_axis = np.array(plane.get('y_axis')[()]).reshape(-1, 1).T
            self.z_axis = np.array(plane.get('z_axis')[()]).reshape(-1, 1).T
            self.transform = np.array(plane.get('transform')[()])
            if isinstance(plane, dict):
                self.shape_name = plane.get('type')
            else:
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
            # Partial derivatives in u and v directions
            deriv = np.zeros((sample_points.shape[0], 3, 2))
            deriv[:, :, 0] = self.x_axis
            deriv[:, :, 1] = self.y_axis
            return deriv
        elif order == 2:
            # Second derivative is zero
            return np.zeros((sample_points.shape[0], 3, 2, 2))
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        normal_vector = np.cross(self.x_axis.squeeze(), self.y_axis.squeeze())
        normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
        normals = np.tile(normal_vector_normalized, (sample_points.shape[0], 1))
        return normals


class Cylinder(Surface):
    """Cylindrical surface."""
    def __init__(self, cylinder, trim_domain=None, transform=None, location=None, radius=None, coefficients=None,
                 x_axis=None, y_axis=None, z_axis=None):
        self.area = -1

        if cylinder is None:
            self.location = location.reshape(-1, 1).T
            self.radius = radius
            self.coefficients = coefficients.reshape(-1, 1).T
            self.trim_domain = trim_domain
            self.x_axis = x_axis.reshape(-1, 1).T
            self.y_axis = y_axis.reshape(-1, 1).T
            self.z_axis = z_axis.reshape(-1, 1).T
            self.transform = transform
            self.shape_name = 'Cylinder'
        else:
            self.location = np.array(cylinder.get('location')[()]).reshape(-1, 1).T
            self.radius = float(cylinder.get('radius')[()])
            self.coefficients = np.array(cylinder.get('coefficients')[()]).reshape(-1, 1).T
            self.trim_domain = np.array(cylinder.get('trim_domain')[()])
            self.x_axis = np.array(cylinder.get('x_axis')[()]).reshape(-1, 1).T
            self.y_axis = np.array(cylinder.get('y_axis')[()]).reshape(-1, 1).T
            self.z_axis = np.array(cylinder.get('z_axis')[()]).reshape(-1, 1).T
            self.transform = np.array(cylinder.get('transform')[()])
            if isinstance(cylinder, dict):
                self.shape_name = cylinder.get('type')
            else:
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
    """Conical surface."""
    def __init__(self, cone, trim_domain=None, transform=None, location=None, radius=None, coefficients=None,
                 apex=None, angle=None, x_axis=None, y_axis=None, z_axis=None):
        self.area = -1

        if cone is None:
            self.location = location.reshape(-1, 1).T
            self.radius = radius
            self.coefficients = coefficients.reshape(-1, 1).T
            self.trim_domain = trim_domain
            self.apex = apex.reshape(-1, 1).T
            self.angle = angle
            self.x_axis = x_axis.reshape(-1, 1).T
            self.y_axis = y_axis.reshape(-1, 1).T
            self.z_axis = z_axis.reshape(-1, 1).T
            self.transform = transform
            self.shape_name = 'Cone'
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
            self.transform = np.array(cone.get('transform')[()])
            if isinstance(cone, dict):
                self.shape_name = cone.get('type')
            else:
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
    """Spherical surface."""
    def __init__(self, sphere, trim_domain=None, transform=None, location=None, radius=None, coefficients=None,
                 x_axis=None, y_axis=None, z_axis=None):
        self.area = -1

        if sphere is None:
            self.location = location.reshape(-1, 1).T
            self.radius = radius
            self.coefficients = coefficients.reshape(-1, 1).T
            self.trim_domain = trim_domain
            self.x_axis = x_axis.reshape(-1, 1).T
            self.y_axis = y_axis.reshape(-1, 1).T
            self.z_axis = z_axis.reshape(-1, 1).T
            self.transform = transform
            self.shape_name = 'Sphere'
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
            self.transform = np.array(sphere.get('transform')[()])
            if isinstance(sphere, dict):
                self.shape_name = sphere.get('type')
            else:
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
        if sample_points.size == 0:
            return np.array([])
        sphere_points = self.sample(sample_points)
        normals = sphere_points - self.location
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
        return normals


class Torus(Surface):
    """Torus (donut-shaped) surface."""
    def __init__(self, torus, trim_domain=None, transform=None, location=None, max_radius=None, min_radius=None,
                 x_axis=None, y_axis=None, z_axis=None):
        self.area = -1

        if torus is None:
            self.location = location.reshape(-1, 1).T
            self.max_radius = max_radius
            self.min_radius = min_radius
            self.trim_domain = trim_domain
            self.x_axis = x_axis.reshape(-1, 1).T
            self.y_axis = y_axis.reshape(-1, 1).T
            self.z_axis = z_axis.reshape(-1, 1).T
            self.transform = transform
            self.shape_name = 'Torus'
        else:
            self.location = np.array(torus.get('location')[()]).reshape(-1, 1).T
            self.max_radius = float(torus.get('max_radius')[()])
            self.min_radius = float(torus.get('min_radius')[()])
            self.trim_domain = np.array(torus.get('trim_domain')[()])
            self.x_axis = np.array(torus.get('x_axis')[()]).reshape(-1, 1).T
            self.y_axis = np.array(torus.get('y_axis')[()]).reshape(-1, 1).T
            self.z_axis = np.array(torus.get('z_axis')[()]).reshape(-1, 1).T
            self.transform = np.array(torus.get('transform')[()])
            if isinstance(torus, dict):
                self.shape_name = torus.get('type')
            else:
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
    """B-spline or NURBS surface (possibly trimmed)."""
    def __init__(self, bspline_surface, trim_domain=None, transform=None, continuity=None, face_domain=None,
                 is_trimmed=None, poles=None, u_closed=None, u_degree=None, u_knots=None, u_rational=None,
                 v_closed=None, v_degree=None, v_knots=None, v_rational=None, weights=None, u_periodic=None,
                 v_periodic=None):
        self.area = -1

        if bspline_surface is None:
            self.continuity = continuity
            self.face_domain = face_domain.reshape(-1, 1).T
            self.is_trimmed = is_trimmed
            self.poles = poles
            self.trim_domain = trim_domain
            self.u_closed = u_closed
            self.u_degree = u_degree
            self.u_knots = u_knots.reshape(-1, 1).T
            self.u_rational = u_rational
            self.v_closed = v_closed
            self.v_degree = v_degree
            self.v_knots = v_knots.reshape(-1, 1).T
            self.v_rational = v_rational
            self.weights = weights.T
            self.u_periodic = u_periodic
            self.v_periodic = v_periodic
            self.transform = transform
            self.shape_name = 'BSpline'
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
            self.weights = np.column_stack(
                [bspline_surface['weights'][str(i)][()] for i in range(len(bspline_surface['weights']))]).T

            self.u_periodic = bool(bspline_surface.get('u_periodic')[()])
            self.v_periodic = bool(bspline_surface.get('v_periodic')[()])
            self.transform = np.array(bspline_surface.get('transform')[()])
            if isinstance(bspline_surface, dict):
                self.shape_name = bspline_surface.get('type')
            else:
                self.shape_name = bspline_surface.get('type')[()].decode('utf8')

        from abs import BSpline
        self.surface_obj = BSpline(
            degree_u = self.u_degree,
            degree_v = self.v_degree,
            u_rational = self.u_rational,
            v_rational = self.v_rational,
            u_knots = self.u_knots.T,
            v_knots = self.v_knots.T,
            grid = self.poles.reshape(-1, 3),
            weights = self.weights.reshape(-1, 1),
            u_periodic = self.u_periodic,
            v_periodic = self.v_periodic
        )


    def sample(self, sample_points):
        if sample_points.size == 0:
            return self.poles[0, 0]
        return np.array(self.surface_obj.sample(sample_points))

    def derivative(self, sample_points, order=1):
        if order == 0:
            return self.sample(sample_points)
        elif order == 1:
            res = np.zeros((sample_points.shape[0], 3, 2))
            res[:, :, 0], res[:, :, 1] = self.surface_obj.first_derivative(sample_points)

            return res
        elif order == 2:
            res = np.zeros((sample_points.shape[0], 3, 2, 2))
            duu, dvv, duv = self.surface_obj.second_derivative(sample_points)
            res[:, :, 0, 0] = duu
            res[:, :, 1, 0] = duv
            res[:, :, 0, 1] = duv
            res[:, :, 1, 1] = dvv

            return res
        else:
            raise ValueError("Order must be 0, 1, or 2")

    def normal(self, sample_points):
        if sample_points.size == 0:
            return np.array([])

        du, dv = self.surface_obj.first_derivative(sample_points)
        normals = np.cross(du, dv)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normal_vectors = normals / norms

        return normal_vectors



class Extrusion(Surface):
    def __init__(self, extrusion, trim_domain=None, transform=None, direction=None, curve=None):
        self.area = -1

        if extrusion is None:
            self.direction = direction.reshape(-1, 1).T
            self.trim_domain = trim_domain
            self.transform = transform
            self.curve = curve
            self.shape_name = 'Extrusion'
        else:
            self.direction = np.array(extrusion.get('direction')[()]).reshape(-1, 1).T
            self.trim_domain = np.array(extrusion.get('trim_domain')[()])
            self.transform = np.array(extrusion.get('transform')[()])
            if isinstance(extrusion, dict):
                self.shape_name = extrusion.get('type')
            else:
                self.shape_name = extrusion.get('type')[()].decode('utf8')
            _, self.curve = create_curve(extrusion['curve'], False)

    def sample(self, points):
        return self.curve.sample(points[:, 0][:, np.newaxis]) + points[:, 1][:, np.newaxis] * self.direction

    def derivative(self, points, order=1):
        if order == 0:
            return self.sample(points)
        elif order == 1:
            res = np.zeros((points.shape[0], 3, 2))
            res[:, :, 0] = self.curve.derivative(points[:, 0][:, np.newaxis], order=1)
            res[:, :, 1] = self.direction
            return res
        elif order == 2:
            res = np.zeros((points.shape[0], 3, 2, 2))
            res[:, :, 0, 0] = self.curve.derivative(points[:, 0][:, np.newaxis], order=2)

        return res


class Revolution(Surface):
    def __init__(self, revolution, trim_domain=None, transform=None, location=None, z_axis=None, curve=None):
        self.area = -1
        if revolution is None:
            self.location = location.reshape(-1, 1).T
            self.trim_domain = trim_domain
            self.transform = transform
            self.z_axis = z_axis.reshape(-1, 1).T
            self.curve = curve
            self.shape_name = 'Revolution'
        else:
            self.location = np.array(revolution.get('location')[()]).reshape(-1, 1).T
            self.trim_domain = np.array(revolution.get('trim_domain')[()])
            self.transform = np.array(revolution.get('transform')[()])
            if isinstance(revolution, dict):
                self.shape_name = revolution.get('type')
            else:
                self.shape_name = revolution.get('type')[()].decode('utf8')
            self.z_axis = np.array(revolution.get('z_axis')[()]).reshape(-1, 1).T
            _, self.curve = create_curve(revolution['curve'], False)

    def sample(self, points):
        dx, dy, dz = self.z_axis[0, 0], self.z_axis[0, 1], self.z_axis[0, 2]

        cos_t = np.cos(points[:, 0])
        sin_t = np.sin(points[:, 0])
        one_minus_cos = 1 - cos_t

        R = np.array([
            [cos_t + dx * dx * one_minus_cos, dx * dy * one_minus_cos - dz * sin_t,
             dx * dz * one_minus_cos + dy * sin_t],
            [dy * dx * one_minus_cos + dz * sin_t, cos_t + dy * dy * one_minus_cos,
             dy * dz * one_minus_cos - dx * sin_t],
            [dz * dx * one_minus_cos - dy * sin_t, dz * dy * one_minus_cos + dx * sin_t,
             cos_t + dz * dz * one_minus_cos]
        ])

        return np.einsum('ijk,ik->ij', R.transpose(2, 0, 1), self.curve.sample(points[:, 1][:, np.newaxis]) - self.location) + self.location

    def derivative(self, points, order=1):
        if order == 0:
            return self.sample(points)
        elif order == 1:
            res = np.zeros((points.shape[0], 3, 2))
            theta = points[:, 0]
            u = self.z_axis[0, :]
            v_param = points[:, 1]
            dx, dy, dz = self.z_axis[0, 0], self.z_axis[0, 1], self.z_axis[0, 2]

            cos_t = np.cos(points[:, 0])
            sin_t = np.sin(points[:, 0])
            one_minus_cos = 1 - cos_t

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # outer product u ⊗ u, shape (3, 3)
            uuT = np.outer(u, u)
            I = np.eye(3)
            u_skew = np.array([
                [0, -u[2], u[1]],
                [u[2], 0, -u[0]],
                [-u[1], u[0], 0]
            ])

            # dR/dtheta, vectorized version for each theta
            dR_dtheta = (
                -sin_t[:, None, None] * I
                + sin_t[:, None, None] * uuT
                + cos_t[:, None, None] * u_skew
            )  # shape (N, 3, 3)

            R = np.array([
                [cos_t + dx * dx * one_minus_cos, dx * dy * one_minus_cos - dz * sin_t,
                 dx * dz * one_minus_cos + dy * sin_t],
                [dy * dx * one_minus_cos + dz * sin_t, cos_t + dy * dy * one_minus_cos,
                 dy * dz * one_minus_cos - dx * sin_t],
                [dz * dx * one_minus_cos - dy * sin_t, dz * dy * one_minus_cos + dx * sin_t,
                 cos_t + dz * dz * one_minus_cos]
            ])

            # v = sampled curve points - self.location
            v = self.curve.sample(v_param[:, None]) - self.location  # shape (N, 3)
            dv_dtheta = np.einsum('nij,nj->ni', dR_dtheta, v)

            # df/dv (i.e., derivative of curve)
            df_dv = self.curve.derivative(v_param[:, None], order=1)  # shape (N, 3)
            dv_dv = np.einsum('nij,nj->ni', R.transpose(2, 0, 1), df_dv)
            res[:, :, 0] = dv_dtheta
            res[:, :, 1] = dv_dv
        elif order == 2:
            res = np.zeros((points.shape[0], 3, 2, 2))

            u = self.z_axis[0,:]

            N = len(points)
            theta = points[:, 0]
            v_param = points[:, 1]

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            one_minus_cos = 1 - cos_t

            # Basic components
            uuT = np.outer(u, u)
            I = np.eye(3)
            u_skew = skew(u)

            # Rotation matrices R
            R = (
                cos_t[:, None, None] * I
                + one_minus_cos[:, None, None] * uuT
                + sin_t[:, None, None] * u_skew
            )  # shape (N, 3, 3)

            # Derivative of R wrt theta
            dR_dtheta = (
                -sin_t[:, None, None] * I
                + sin_t[:, None, None] * uuT
                + cos_t[:, None, None] * u_skew
            )  # shape (N, 3, 3)

            # Second derivative of R wrt theta
            d2R_dtheta2 = (
                -cos_t[:, None, None] * I
                + cos_t[:, None, None] * uuT
                - sin_t[:, None, None] * u_skew
            )  # shape (N, 3, 3)

            # Sample and differentiate the curve
            f = self.curve.sample(v_param[:, None]) - self.location  # shape (N, 3)
            df_dv = self.curve.derivative(v_param[:, None], order=2)  # shape (N, 3)
            d2f_dv2 = self.curve.derivative(v_param[:, None], order=2)  # shape (N, 3)


            # Second derivative outputs
            d2y_dtheta2 = np.einsum('nij,nj->ni', d2R_dtheta2, f)
            d2y_dthetadv = np.einsum('nij,nj->ni', dR_dtheta, df_dv)
            d2y_dv2 = np.einsum('nij,nj->ni', R, d2f_dv2)

            res[:, :, 0, 0] = d2y_dtheta2
            res[:, :, 0, 1] = d2y_dthetadv
            res[:, :, 1, 0] = d2y_dthetadv
            res[:, :, 1, 1] = d2y_dv2

        return res


class Offset(Surface):
    def __init__(self, offset, trim_domain=None, transform=None, value=None, surface=None):
        self.area = -1

        if offset is None:
            self.trim_domain = trim_domain
            self.transform = transform
            self.value = value
            self.surface = surface
            self.shape_name = 'Offset'
        else:
            self.trim_domain = np.array(offset.get('trim_domain')[()])
            self.transform = np.array(offset.get('transform')[()])
            self.value = np.float64(offset.get('value')[()])
            if isinstance(offset, dict):
                self.shape_name = offset.get('type')
            else:
                self.shape_name = offset.get('type')[()].decode('utf8')
            _, self.surface = create_surface(offset['surface'], False)

    def sample(self, points):

        N = points.shape[0]
        base_points = self.surface.sample(points)
        normals = self.surface.normal(points)
        offset_points = base_points + self.value * normals

        return offset_points

    def derivative(self, points, order=1):
        if order == 0:
            return self.sample(points)
        elif order == 1:
            diff = self.surface.derivative(points, order=1)

            res = diff.copy()

            du = diff[:, :, 0]
            dv = diff[:, :, 1]

            diffdiff = self.surface.derivative(points, order=2)

            c = np.cross(du, dv) # diff[:, :, 0] x diff[:, :, 1]
            norm = np.linalg.norm(c, axis=1)[:, None]
            dcdu = np.cross(diffdiff[:, :, 0, 0], diff[:, :, 1]) + np.cross(diff[:, :, 0], diffdiff[: ,:, 1, 0])
            dcdv = np.cross(diffdiff[:, :, 0, 1], diff[:, :, 1]) + np.cross(diff[:, :, 0], diffdiff[:, :, 1, 1])

            res[:, :, 0] += self.value * (dcdu/norm - c @ (c.T @ dcdu)/norm**3)
            res[:, :, 1] += self.value * (dcdv/norm - c @ (c.T @ dcdv)/norm**3)

            return res

