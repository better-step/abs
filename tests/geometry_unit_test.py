from tests.test_utilities import *
import unittest
import numpy as np
import warnings


def surface_derivative(surface, sample_points):
    epsilon = 1e-6

    # testing ds/du order 1
    sample_points_plus = sample_points.copy()
    sample_points_plus[:, 0] += epsilon
    deriv = (surface.sample(sample_points_plus) - surface.sample(sample_points)) / epsilon
    deriv1 = surface.derivative(sample_points, 1)
    p = np.abs(deriv - deriv1[:, :, 0]).max()

    # testing ds/dv order 1
    sample_points_plus = sample_points.copy()
    sample_points_plus[:, 1] += epsilon
    deriv = (surface.sample(sample_points_plus) - surface.sample(sample_points)) / epsilon
    deriv1 = surface.derivative(sample_points, 1)
    q = np.abs(deriv - deriv1[:, :, 1]).max()

    # testing ds/du order 2
    sample_points_plus = sample_points.copy()
    sample_points_plus[:, 0] += epsilon
    deriv = (surface.derivative(sample_points_plus, 1) - surface.derivative(sample_points, 1)) / epsilon
    deriv1 = surface.derivative(sample_points, 2)
    p2 = np.abs(deriv - deriv1[:, :, :, 0]).max()

    # testing ds/dv order 2
    sample_points_plus = sample_points.copy()
    sample_points_plus[:, 1] += epsilon
    deriv = (surface.derivative(sample_points_plus, 1) - surface.derivative(sample_points, 1)) / epsilon
    deriv1 = surface.derivative(sample_points, 2)
    q2 = np.abs(deriv - deriv1[:, :, :, 1]).max()

    return p, q, p2, q2


def curves_derivative(curve, sample_points):
    epsilon = (sample_points[1] - sample_points[0]) * 1e-5
    sample_points_plus = sample_points.copy()
    sample_points_plus += epsilon
    deriv = (curve.sample(sample_points_plus) - curve.sample(sample_points)) / epsilon
    deriv1 = curve.derivative(sample_points, 1)
    p = np.abs(deriv - deriv1).max()

    sample_points_plus = sample_points.copy()
    sample_points_plus += epsilon
    deriv = (curve.derivative(sample_points_plus, 1) - curve.derivative(sample_points, 1)) / epsilon
    deriv1 = curve.derivative(sample_points, 2)
    q = np.abs(deriv - deriv1).max()

    return p, q


def generate_points_on_curve(curve, num_samples=1000):
    param_range = np.linspace(curve._interval[0, 0], curve._interval[0, 1], num_samples)
    param_range = param_range[:, None]
    points = curve.sample(param_range.reshape(-1, 1))
    return param_range, points


def estimate_normal(curve, num_samples=1000):
    _, points = generate_points_on_curve(curve, num_samples)
    lines = np.diff(points, axis=0)
    lengths = np.linalg.norm(lines, axis=1)
    normalized_lines = lines / lengths[:, np.newaxis]
    rotation_matrix = np.array([[0, -1], [1, 0]])
    rotated_p = normalized_lines @ rotation_matrix.T
    return rotated_p


class Geometrytest(unittest.TestCase):

    def test_line2d(self):
        line = test_line2d()
        self.assertEqual(line._location.shape, (1, 2))
        self.assertEqual(line._direction.shape, (1, 2))
        self.assertEqual(line._interval.shape, (1, 2))
        sample_points = np.linspace(0, 1, 10).reshape(-1, 1)

        # sampling
        self.assertEqual(line.sample(sample_points).shape, (10, 2))

        # derivative
        self.assertEqual(line.derivative(sample_points, 0).shape, (10, 2))
        self.assertEqual(line.derivative(sample_points, 1).shape, (10, 2))
        self.assertEqual(line.derivative(sample_points, 2).shape, (10, 2))
        d, d2 = curves_derivative(line, sample_points)
        self.assertTrue(d < 1e-7)
        self.assertTrue(d2 < 1e-7)

        # length
        num_samples = 1000  # Can be adjusted for precision
        param_points, points = generate_points_on_curve(line, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) -
                            (line._length if line._length != -1 else line.length()) < 1e-4))

        # normals
        rotated_p = estimate_normal(line, num_samples)
        self.assertTrue(abs(np.sum(rotated_p - line.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        self.assertTrue(np.allclose(np.linalg.norm(line.normal(param_points), axis=1), 1, atol=1e-8))

    def test_circle2d(self):
        circle = test_circle2d()
        self.assertEqual(circle._location.shape, (1, 2))
        self.assertEqual(type(circle._radius), float)
        self.assertEqual(circle._interval.shape, (1, 2))
        self.assertEqual(circle._x_axis.shape, (1, 2))
        self.assertEqual(circle._y_axis.shape, (1, 2))

        # sampling
        sample_points = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        self.assertEqual(circle.sample(sample_points).shape, (10, 2))

        # derivative
        self.assertEqual(circle.derivative(sample_points, 0).shape, (10, 2))
        self.assertEqual(circle.derivative(sample_points, 1).shape, (10, 2))
        self.assertEqual(circle.derivative(sample_points, 2).shape, (10, 2))
        self.assertEqual(circle.derivative(sample_points, 3).shape, (10, 2))
        self.assertEqual(circle.derivative(sample_points, 4).shape, (10, 2))
        d, d2 = curves_derivative(circle, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

        # length
        num_samples = 1000
        param_points, points = generate_points_on_curve(circle, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) -
                            (circle._length if circle._length != -1 else circle.length()) < 1e-4))

        # normals
        rotated_p = estimate_normal(circle, num_samples)
        self.assertTrue(abs(np.sum(rotated_p - circle.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        self.assertTrue(np.allclose(np.linalg.norm(circle.normal(param_points), axis=1), 1, atol=1e-8))

    def test_ellipse2d(self):
        ellipse = test_ellipse2d()
        self.assertEqual(ellipse._focus1.shape, (1, 2))
        self.assertEqual(ellipse._focus2.shape, (1, 2))
        self.assertEqual(ellipse._interval.shape, (1, 2))
        self.assertEqual(type(ellipse._maj_radius), float)
        self.assertEqual(type(ellipse._min_radius), float)
        self.assertEqual(ellipse._x_axis.shape, (1, 2))
        self.assertEqual(ellipse._y_axis.shape, (1, 2))
        sample_points = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        self.assertEqual(ellipse.sample(sample_points).shape, (10, 2))

        # derivative
        self.assertEqual(ellipse.derivative(sample_points, 0).shape, (10, 2))
        self.assertEqual(ellipse.derivative(sample_points, 1).shape, (10, 2))
        self.assertEqual(ellipse.derivative(sample_points, 2).shape, (10, 2))
        self.assertEqual(ellipse.derivative(sample_points, 3).shape, (10, 2))
        self.assertEqual(ellipse.derivative(sample_points, 4).shape, (10, 2))
        d, d2 = curves_derivative(ellipse, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

        # length
        num_samples = 1000  # Can be adjusted for precision
        param_points, points = generate_points_on_curve(ellipse, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) -
                            (ellipse._length if ellipse._length != -1 else ellipse.length()) < 1e-4))

        # normals
        rotated_p = estimate_normal(ellipse, num_samples)
        self.assertTrue(abs(np.sum(rotated_p - ellipse.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        self.assertTrue(np.allclose(np.linalg.norm(ellipse.normal(param_points), axis=1), 1, atol=1e-8))

    def test_bspline_curve2d(self):
        bspline_curve2d = test_bspline_curve2d()
        self.assertEqual(bspline_curve2d._closed, False)
        self.assertEqual(type(bspline_curve2d._continuity), int)
        self.assertEqual(type(bspline_curve2d._degree), int)
        self.assertEqual(bspline_curve2d._interval.shape, (1, 2))
        self.assertEqual(bspline_curve2d._knots.shape[0], 1)
        self.assertEqual(bspline_curve2d._poles.shape[1], 2)
        self.assertEqual(type(bspline_curve2d._rational), bool)
        self.assertEqual(bspline_curve2d._weights.shape[1], 1)
        self.assertEqual(bspline_curve2d._poles.shape[0], bspline_curve2d._weights.shape[0])

        # sample points
        umin_value, umax_value = bspline_curve2d._interval.T
        gridX = np.linspace(umin_value, umax_value)
        sample_points = gridX.reshape(-1, 1)
        self.assertEqual(bspline_curve2d.sample(sample_points).shape, (gridX.shape[0], 2))

        self.assertEqual(bspline_curve2d.derivative(sample_points, 0).shape, (50, 2))
        self.assertEqual(bspline_curve2d.derivative(sample_points, 1).shape, (50, 2))
        d, d2 = curves_derivative(bspline_curve2d, sample_points)
        self.assertTrue(d < 1e-4)
        # self.assertTrue(d2 < 1e-4)

        # length
        num_samples = 1000
        param_points, points = generate_points_on_curve(bspline_curve2d, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - bspline_curve2d.length() < 1e-4))

        # normals
        rotated_p = estimate_normal(bspline_curve2d, num_samples)
        self.assertTrue(abs(np.sum(rotated_p - bspline_curve2d.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        self.assertTrue(np.allclose(np.linalg.norm(bspline_curve2d.normal(param_points), axis=1), 1, atol=1e-8))

    def test_line3d(self):
        line = test_line3d()
        self.assertEqual(line._location.shape, (1, 3))
        self.assertEqual(line._direction.shape, (1, 3))
        self.assertEqual(line._interval.shape, (1, 2))
        sample_points: None = np.linspace(0, 1, 10).reshape(-1, 1)

        # sampling
        self.assertEqual(line.sample(sample_points).shape, (10, 3))

        # derivative shape
        self.assertEqual(line.derivative(sample_points, 0).shape, (10, 3))
        self.assertEqual(line.derivative(sample_points, 1).shape, (10, 3))
        self.assertEqual(line.derivative(sample_points, 2).shape, (10, 3))

        d, d2 = curves_derivative(line, sample_points)
        self.assertTrue(d < 1e-7)
        self.assertTrue(d2 < 1e-7)

    def test_circle3d(self):
        circle = test_circle3d()
        self.assertEqual(circle._location.shape, (1, 3))
        self.assertEqual(type(circle._radius), float)
        self.assertEqual(circle._interval.shape, (1, 2))
        self.assertEqual(circle._x_axis.shape, (1, 3))
        self.assertEqual(circle._y_axis.shape, (1, 3))
        self.assertEqual(circle._z_axis.shape, (1, 3))
        sample_points = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        self.assertEqual(circle.sample(sample_points).shape, (10, 3))
        # derivative
        self.assertEqual(circle.derivative(sample_points, 0).shape, (10, 3))
        self.assertEqual(circle.derivative(sample_points, 1).shape, (10, 3))
        self.assertEqual(circle.derivative(sample_points, 2).shape, (10, 3))
        self.assertEqual(circle.derivative(sample_points, 3).shape, (10, 3))
        self.assertEqual(circle.derivative(sample_points, 4).shape, (10, 3))

        d, d2 = curves_derivative(circle, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

    def test_ellipse3d(self):
        ellipse = test_ellipse3d()
        self.assertEqual(ellipse._focus1.shape, (1, 3))
        self.assertEqual(ellipse._focus2.shape, (1, 3))
        self.assertEqual(ellipse._interval.shape, (1, 2))
        self.assertEqual(type(ellipse._maj_radius), float)
        self.assertEqual(type(ellipse._min_radius), float)
        self.assertEqual(ellipse._x_axis.shape, (1, 3))
        self.assertEqual(ellipse._y_axis.shape, (1, 3))
        self.assertEqual(ellipse._z_axis.shape, (1, 3))
        sample_points = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        self.assertEqual(ellipse.sample(sample_points).shape, (10, 3))
        # derivative
        self.assertEqual(ellipse.derivative(sample_points, 0).shape, (10, 3))
        self.assertEqual(ellipse.derivative(sample_points, 1).shape, (10, 3))
        self.assertEqual(ellipse.derivative(sample_points, 2).shape, (10, 3))
        self.assertEqual(ellipse.derivative(sample_points, 3).shape, (10, 3))
        self.assertEqual(ellipse.derivative(sample_points, 4).shape, (10, 3))

        d, d2 = curves_derivative(ellipse, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

    def test_bspline_curve3d(self):
        bspline_curve3d = test_bspline_curve3d()
        self.assertEqual(type(bspline_curve3d._closed), bool)
        self.assertEqual(type(bspline_curve3d._continuity), int)
        self.assertEqual(type(bspline_curve3d._degree), int)
        self.assertEqual(bspline_curve3d._interval.shape, (1, 2))
        self.assertEqual(bspline_curve3d._knots.shape[0], 1)
        self.assertEqual(bspline_curve3d._poles.shape[1], 3)
        self.assertEqual(type(bspline_curve3d._rational), bool)
        self.assertEqual(bspline_curve3d._weights.shape[1], 1)

        # sample points
        umin_value, umax_value = bspline_curve3d._interval.T
        gridX = np.linspace(umin_value, umax_value)
        sample_points = gridX.reshape(-1, 1)
        self.assertEqual(bspline_curve3d.sample(sample_points).shape, (gridX.shape[0], 3))

        # derivative
        self.assertEqual(bspline_curve3d.derivative(sample_points, 0).shape, (50, 3))
        self.assertEqual(bspline_curve3d.derivative(sample_points, 1).shape, (50, 3))
        d, d2 = curves_derivative(bspline_curve3d, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

    def test_plane(self):
        plane = test_plane()
        self.assertEqual(plane._coefficients.shape, (1, 4))
        self.assertEqual(plane._location.shape, (1, 3))
        self.assertEqual(plane._trim_domain.shape, (2, 2))
        self.assertEqual(plane._x_axis.shape, (1, 3))
        self.assertEqual(plane._y_axis.shape, (1, 3))
        self.assertEqual(plane._z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = plane._trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        gridX = gridX.reshape((np.prod(gridX.shape),)).reshape(-1, 1)
        gridY = gridY.reshape((np.prod(gridY.shape),)).reshape(-1, 1)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)

        self.assertEqual(plane.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        # derivative:
        p, q, du, dv = surface_derivative(plane, sample_points)
        self.assertTrue(p < 1e-7)
        self.assertTrue(q < 1e-7)
        self.assertTrue(du < 1e-7)
        self.assertTrue(dv < 1e-7)

        self.assertEqual(plane.area(), 1)

    def test_cylinder(self):
        cylinder = test_cylinder()
        self.assertEqual(cylinder._location.shape, (1, 3))
        self.assertEqual(type(cylinder._radius), float)
        self.assertEqual(cylinder._coefficients.shape, (1, 10))
        self.assertEqual(cylinder._trim_domain.shape, (2, 2))
        self.assertEqual(cylinder._x_axis.shape, (1, 3))
        self.assertEqual(cylinder._y_axis.shape, (1, 3))
        self.assertEqual(cylinder._z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = cylinder._trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(cylinder.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        du, dv, d2u, d2v = surface_derivative(cylinder, sample_points)
        self.assertTrue(du < 1e-6)
        self.assertTrue(dv < 1e-6)
        self.assertTrue(d2u < 1e-6)
        self.assertTrue(d2v < 1e-6)

        self.assertEqual(cylinder.area(), 1)

    def test_cone(self):
        cone = test_cone()
        self.assertEqual(type(cone._angle), float)
        self.assertEqual(cone._apex.shape, (1, 3))
        self.assertEqual(cone._coefficients.shape, (1, 10))
        self.assertEqual(cone._location.shape, (1, 3))
        self.assertEqual(type(cone._radius), float)
        self.assertEqual(cone._trim_domain.shape, (2, 2))
        self.assertEqual(cone._x_axis.shape, (1, 3))
        self.assertEqual(cone._y_axis.shape, (1, 3))
        self.assertEqual(cone._z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = cone._trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(cone.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        du, dv, d2u, d2v = surface_derivative(cone, sample_points)
        self.assertTrue(du < 1e-6)
        self.assertTrue(dv < 1e-6)
        self.assertTrue(d2u < 1e-6)
        self.assertTrue(d2v < 1e-6)


    def test_sphere(self):
        sphere = test_sphere()
        self.assertEqual(sphere._coefficients.shape, (1, 10))
        self.assertEqual(sphere._location.shape, (1, 3))
        self.assertEqual(type(sphere._radius), float)
        self.assertEqual(sphere._trim_domain.shape, (2, 2))
        self.assertEqual(sphere._x_axis.shape, (1, 3))
        self.assertEqual(sphere._y_axis.shape, (1, 3))
        self.assertEqual(sphere._z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = sphere._trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(sphere.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        # derivative: du, dv
        du, dv, d2u, d2v = surface_derivative(sphere, sample_points)
        self.assertTrue(du < 1e-6)
        self.assertTrue(dv < 1e-6)
        self.assertTrue(d2u < 1e-6)
        self.assertTrue(d2v < 1e-6)

    def test_torus(self):
        torus = test_torus()
        self.assertEqual(torus._location.shape, (1, 3))
        self.assertEqual(type(torus._max_radius), float)
        self.assertEqual(type(torus._min_radius), float)
        self.assertEqual(torus._trim_domain.shape, (2, 2))
        self.assertEqual(torus._x_axis.shape, (1, 3))
        self.assertEqual(torus._y_axis.shape, (1, 3))
        self.assertEqual(torus._z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = torus._trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(torus.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        du, dv, d2u, d2v = surface_derivative(torus, sample_points)
        self.assertTrue(du < 1e-6)
        self.assertTrue(dv < 1e-6)
        self.assertTrue(d2u < 1e-6)
        self.assertTrue(d2v < 1e-6)

    def test_bspline_surface(self):
        bspline_surface = test_bspline_surface()

        x = bspline_surface.area()
        self.assertEqual(type(bspline_surface._continuity), int)
        self.assertEqual(bspline_surface._face_domain.shape, (1, 4))
        self.assertEqual(type(bspline_surface._is_trimmed), bool)
        self.assertEqual(bspline_surface._poles.shape[2], 3)
        self.assertEqual(bspline_surface._trim_domain.shape, (2, 2))
        self.assertEqual(type(bspline_surface._u_closed), bool)
        self.assertEqual(type(bspline_surface._u_degree), int)
        self.assertEqual(bspline_surface._u_knots.shape[0], 1)
        self.assertEqual(type(bspline_surface._u_rational), bool)
        self.assertEqual(type(bspline_surface._v_closed), bool)
        self.assertEqual(type(bspline_surface._v_degree), int)
        self.assertEqual(bspline_surface._v_knots.shape[0], 1)
        self.assertEqual(type(bspline_surface._v_rational), bool)

        # TODO: check weights, there is inconsistency when initializing weights

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = bspline_surface._trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(bspline_surface.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        # derivative
        du, dv, d2u, d2v = surface_derivative(bspline_surface, sample_points)
        self.assertTrue(du < 1e-4)
        self.assertTrue(dv < 1e-4)
        self.assertTrue(d2u < 1e-4)
        self.assertTrue(d2v < 1e-4)


if __name__ == '__main__':
    unittest.main()
