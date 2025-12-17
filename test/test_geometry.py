import unittest
from random import sample
import numpy as np
from .test_utilities import *
from scipy.interpolate import BSpline

def surface_derivative(surface, sample_points, epsilon=1e-6):
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
    p = np.abs(deriv - deriv1).max()/max(1, np.abs(np.mean(deriv)))

    sample_points_plus = sample_points.copy()
    sample_points_plus += epsilon
    deriv = (curve.derivative(sample_points_plus, 1) - curve.derivative(sample_points, 1)) / epsilon
    deriv1 = curve.derivative(sample_points, 2)
    q = np.abs(deriv - deriv1).max()/max(1, np.abs(np.mean(deriv)))

    return p, q


def generate_points_on_curve(curve, num_samples=1000):
    param_range = np.linspace(curve.interval[0, 0], curve.interval[0, 1], num_samples)
    param_range = param_range[:, None]
    points = curve.sample(param_range.reshape(-1, 1))
    return param_range, points


# def estimate_normal(curve, num_samples=1000):

#     _, points = generate_points_on_curve(curve, num_samples)

#     if points.shape[1] == 3:
#         tangents = np.gradient(points, axis=0)
#         tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
#         dT = np.gradient(tangents, axis=0)
#         norms = np.linalg.norm(dT, axis=1, keepdims=True)
#         norms[norms == 0] = 1.0
#         normals = dT / norms
#         return normals
#     else:
#         lines = np.diff(points, axis=0)
#         lengths = np.linalg.norm(lines, axis=1)
#         normalized_lines = lines / lengths[:, np.newaxis]
#         rotation_matrix = np.array([[0, -1], [1, 0]])
#         rotated_p = normalized_lines @ rotation_matrix.T
#         return rotated_p


class TestGeometry(unittest.TestCase):

    def test_shapes(self):
        line3d()
        line2d()

        circle3d()
        circle2d()

        ellipse3d()
        ellipse2d()

        bspline_curve3d()
        bspline_curve2d()

        plane()
        cylinder()
        cone()
        sphere()
        torus()
        bspline_surface()

    def test_line2d(self):

        shape = line2d()
        self.assertEqual(shape.location.shape, (1, 2))
        self.assertEqual(shape.direction.shape, (1, 2))
        self.assertEqual(shape.interval.shape, (1, 2))
        sample_points = np.linspace(0, 1, 10).reshape(-1, 1)

        # sampling
        self.assertEqual(shape.sample(sample_points).shape, (10, 2))

        # derivative
        self.assertEqual(shape.derivative(sample_points, 0).shape, (10, 2))
        self.assertEqual(shape.derivative(sample_points, 1).shape, (10, 2))
        self.assertEqual(shape.derivative(sample_points, 2).shape, (10, 2))
        d, d2 = curves_derivative(shape, sample_points)
        self.assertTrue(d < 1e-7)
        self.assertTrue(d2 < 1e-7)

        # length
        num_samples = 1000
        param_points, points = generate_points_on_curve(shape, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) -
                            (shape.length if shape.length != -1 else shape.get_length()) < 1e-4))

        # normals
        # rotated_p = estimate_normal(shape, num_samples)
        # self.assertTrue(abs(np.sum(rotated_p - shape.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        # self.assertTrue(np.allclose(np.linalg.norm(shape.normal(param_points), axis=1), 1, atol=1e-8))

    def test_circle2d(self):
        shape = circle2d()
        self.assertEqual(shape.location.shape, (1, 2))
        self.assertEqual(type(shape.radius), float)
        self.assertEqual(shape.interval.shape, (1, 2))
        self.assertEqual(shape.x_axis.shape, (1, 2))
        self.assertEqual(shape.y_axis.shape, (1, 2))

        # sampling
        sample_points = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        self.assertEqual(shape.sample(sample_points).shape, (10, 2))

        # derivative
        self.assertEqual(shape.derivative(sample_points, 0).shape, (10, 2))
        self.assertEqual(shape.derivative(sample_points, 1).shape, (10, 2))
        self.assertEqual(shape.derivative(sample_points, 2).shape, (10, 2))
        self.assertEqual(shape.derivative(sample_points, 3).shape, (10, 2))
        self.assertEqual(shape.derivative(sample_points, 4).shape, (10, 2))
        d, d2 = curves_derivative(shape, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

        # length
        num_samples = 1000
        param_points, points = generate_points_on_curve(shape, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) -
                            (shape.length if shape.length != -1 else shape.get_length()) < 1e-4))

        # normals
        # rotated_p = estimate_normal(shape, num_samples)
        # self.assertTrue(abs(np.sum(rotated_p - shape.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        # self.assertTrue(np.allclose(np.linalg.norm(shape.normal(param_points), axis=1), 1, atol=1e-8))

    def test_ellipse2d(self):

        ellipse = ellipse2d()
        self.assertEqual(ellipse.focus1.shape, (1, 2))
        self.assertEqual(ellipse.focus2.shape, (1, 2))
        self.assertEqual(ellipse.interval.shape, (1, 2))
        self.assertEqual(type(ellipse.maj_radius), float)
        self.assertEqual(type(ellipse.min_radius), float)
        self.assertEqual(ellipse.x_axis.shape, (1, 2))
        self.assertEqual(ellipse.y_axis.shape, (1, 2))
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
                            (ellipse.length if ellipse.length != -1 else ellipse.get_length()) < 1e-4))

        # normals
        # rotated_p = estimate_normal(ellipse, num_samples)
        # self.assertTrue(abs(np.sum(rotated_p - ellipse.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        # self.assertTrue(np.allclose(np.linalg.norm(ellipse.normal(param_points), axis=1), 1, atol=1e-8))

    def test_bspline_curve2d(self):

        shape = bspline_curve2d()

        self.assertEqual(shape.closed, False)
        self.assertEqual(type(shape.continuity), int)
        self.assertEqual(type(shape.degree), int)
        self.assertEqual(shape.interval.shape, (1, 2))
        self.assertEqual(shape.knots.shape[0], 1)
        self.assertEqual(shape.poles.shape[1], 2)
        self.assertEqual(type(shape.rational), bool)
        self.assertEqual(shape.weights.shape[1], 1)
        self.assertEqual(shape.poles.shape[0], shape.weights.shape[0])

        # sample points
        umin_value, umax_value = shape.interval.T
        gridX = np.linspace(umin_value, umax_value)
        sample_points = gridX.reshape(-1, 1)

        self.assertEqual(shape.sample(sample_points).shape, (gridX.shape[0], 2))
        self.assertEqual(shape.derivative(sample_points, 0).shape, (50, 2))
        self.assertEqual(shape.derivative(sample_points, 1).shape, (50, 2))

        d, d2 = curves_derivative(shape, sample_points)

        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

        # length
        num_samples = 1000
        param_points, points = generate_points_on_curve(shape, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - shape.get_length() < 1e-4))

        # normals
        # rotated_p = estimate_normal(shape, num_samples)
        # self.assertTrue(abs(np.sum(rotated_p - shape.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        # self.assertTrue(np.allclose(np.linalg.norm(shape.normal(param_points), axis=1), 1, atol=1e-8))



    def test_line3d(self):
        shape = line3d()
        self.assertEqual(shape.location.shape, (1, 3))
        self.assertEqual(shape.direction.shape, (1, 3))
        self.assertEqual(shape.interval.shape, (1, 2))
        sample_points: None = np.linspace(0, 1, 10).reshape(-1, 1)

        # sampling
        self.assertEqual(shape.sample(sample_points).shape, (10, 3))

        # derivative shape
        self.assertEqual(shape.derivative(sample_points, 0).shape, (10, 3))
        self.assertEqual(shape.derivative(sample_points, 1).shape, (10, 3))
        self.assertEqual(shape.derivative(sample_points, 2).shape, (10, 3))

        d, d2 = curves_derivative(shape, sample_points)
        self.assertTrue(d < 1e-7)
        self.assertTrue(d2 < 1e-7)

        # length
        num_samples = 1000
        param_points, points = generate_points_on_curve(shape, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - shape.get_length() < 1e-4))

    def test_circle3d(self):
        shape = circle3d()
        self.assertEqual(shape.location.shape, (1, 3))
        self.assertEqual(type(shape.radius), float)
        self.assertEqual(shape.interval.shape, (1, 2))
        self.assertEqual(shape.x_axis.shape, (1, 3))
        self.assertEqual(shape.y_axis.shape, (1, 3))
        self.assertEqual(shape.z_axis.shape, (1, 3))
        sample_points = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        self.assertEqual(shape.sample(sample_points).shape, (10, 3))

        # derivative
        self.assertEqual(shape.derivative(sample_points, 0).shape, (10, 3))
        self.assertEqual(shape.derivative(sample_points, 1).shape, (10, 3))
        self.assertEqual(shape.derivative(sample_points, 2).shape, (10, 3))
        self.assertEqual(shape.derivative(sample_points, 3).shape, (10, 3))
        self.assertEqual(shape.derivative(sample_points, 4).shape, (10, 3))

        d, d2 = curves_derivative(shape, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

        # length
        num_samples = 1000
        param_points, points = generate_points_on_curve(shape, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - shape.get_length() < 1e-4))

    def test_ellipse3d(self):
        ellipse = ellipse3d()
        self.assertEqual(ellipse.focus1.shape, (1, 3))
        self.assertEqual(ellipse.focus2.shape, (1, 3))
        self.assertEqual(ellipse.interval.shape, (1, 2))
        self.assertEqual(type(ellipse.maj_radius), float)
        self.assertEqual(type(ellipse.min_radius), float)
        self.assertEqual(ellipse.x_axis.shape, (1, 3))
        self.assertEqual(ellipse.y_axis.shape, (1, 3))
        self.assertEqual(ellipse.z_axis.shape, (1, 3))
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

        # length
        num_samples = 1000
        param_points, points = generate_points_on_curve(ellipse, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - ellipse.get_length() < 1e-4))

        #normals
        # rotated_p = estimate_normal(ellipse, num_samples)

    def test_bspline_curve3d(self):
        shape = bspline_curve3d()
        self.assertEqual(type(shape.closed), bool)
        self.assertEqual(type(shape.continuity), int)
        self.assertEqual(type(shape.degree), int)
        self.assertEqual(shape.interval.shape, (1, 2))
        self.assertEqual(shape.knots.shape[0], 1)
        self.assertEqual(shape.poles.shape[1], 3)
        self.assertEqual(type(shape.rational), bool)
        self.assertEqual(shape.weights.shape[1], 1)

        # sample points
        umin_value, umax_value = shape.interval.T
        gridX = np.linspace(umin_value, umax_value)
        sample_points = gridX.reshape(-1, 1)
        self.assertEqual(shape.sample(sample_points).shape, (gridX.shape[0], 3))

        # derivative
        self.assertEqual(shape.derivative(sample_points, 0).shape, (50, 3))
        self.assertEqual(shape.derivative(sample_points, 1).shape, (50, 3))
        d, d2 = curves_derivative(shape, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

        # length
        num_samples = 1000
        param_points, points = generate_points_on_curve(shape, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - shape.get_length() < 1e-4))


    def test_plane(self):
        shape = plane()
        self.assertEqual(shape.coefficients.shape, (1, 4))
        self.assertEqual(shape.location.shape, (1, 3))
        self.assertEqual(shape.trim_domain.shape, (2, 2))
        self.assertEqual(shape.x_axis.shape, (1, 3))
        self.assertEqual(shape.y_axis.shape, (1, 3))
        self.assertEqual(shape.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = shape.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        gridX = gridX.reshape((np.prod(gridX.shape),)).reshape(-1, 1)
        gridY = gridY.reshape((np.prod(gridY.shape),)).reshape(-1, 1)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)

        self.assertEqual(shape.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        # derivative:
        p, q, du, dv = surface_derivative(shape, sample_points)
        self.assertTrue(p < 1e-7)
        self.assertTrue(q < 1e-7)
        self.assertTrue(du < 1e-7)
        self.assertTrue(dv < 1e-7)

        # self.assertEqual(int(shape.get_area()), 1)
        self.assertTrue(abs(shape.get_area() - 1) < 1e-7)

    def test_cylinder(self):
        shape = cylinder()
        self.assertEqual(shape.location.shape, (1, 3))
        self.assertEqual(type(shape.radius), float)
        self.assertEqual(shape.coefficients.shape, (1, 10))
        self.assertEqual(shape.trim_domain.shape, (2, 2))
        self.assertEqual(shape.x_axis.shape, (1, 3))
        self.assertEqual(shape.y_axis.shape, (1, 3))
        self.assertEqual(shape.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = shape.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(shape.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        du, dv, d2u, d2v = surface_derivative(shape, sample_points)
        self.assertTrue(du < 1e-6)
        self.assertTrue(dv < 1e-6)
        self.assertTrue(d2u < 1e-6)
        self.assertTrue(d2v < 1e-6)

        self.assertTrue(abs(shape.get_area() - 1) < 1e-7)

    def test_cone(self):
        shape = cone()
        self.assertEqual(type(shape.angle), float)
        self.assertEqual(shape.apex.shape, (1, 3))
        self.assertEqual(shape.coefficients.shape, (1, 10))
        self.assertEqual(shape.location.shape, (1, 3))
        self.assertEqual(type(shape.radius), float)
        self.assertEqual(shape.trim_domain.shape, (2, 2))
        self.assertEqual(shape.x_axis.shape, (1, 3))
        self.assertEqual(shape.y_axis.shape, (1, 3))
        self.assertEqual(shape.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = shape.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(shape.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        du, dv, d2u, d2v = surface_derivative(shape, sample_points)
        self.assertTrue(du < 1e-6)
        self.assertTrue(dv < 1e-6)
        self.assertTrue(d2u < 1e-6)
        self.assertTrue(d2v < 1e-6)


    def test_sphere(self):
        shape = sphere()
        self.assertEqual(shape.coefficients.shape, (1, 10))
        self.assertEqual(shape.location.shape, (1, 3))
        self.assertEqual(type(shape.radius), float)
        self.assertEqual(shape.trim_domain.shape, (2, 2))
        self.assertEqual(shape.x_axis.shape, (1, 3))
        self.assertEqual(shape.y_axis.shape, (1, 3))
        self.assertEqual(shape.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = shape.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(shape.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        # derivative: du, dv
        du, dv, d2u, d2v = surface_derivative(shape, sample_points)
        self.assertTrue(du < 1e-6)
        self.assertTrue(dv < 1e-6)
        self.assertTrue(d2u < 1e-6)
        self.assertTrue(d2v < 1e-6)

    def test_torus(self):
        shape = torus()
        self.assertEqual(shape.location.shape, (1, 3))
        self.assertEqual(type(shape.max_radius), float)
        self.assertEqual(type(shape.min_radius), float)
        self.assertEqual(shape.trim_domain.shape, (2, 2))
        self.assertEqual(shape.x_axis.shape, (1, 3))
        self.assertEqual(shape.y_axis.shape, (1, 3))
        self.assertEqual(shape.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = shape.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(shape.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        du, dv, d2u, d2v = surface_derivative(shape, sample_points)
        self.assertTrue(du < 1e-6)
        self.assertTrue(dv < 1e-6)
        self.assertTrue(d2u < 1e-6)
        self.assertTrue(d2v < 1e-6)

    def test_bspline_surface(self):
        shape, fast = bspline_surface()

        self.assertEqual(type(shape.continuity), int)
        self.assertEqual(shape.face_domain.shape, (1, 4))
        self.assertEqual(type(shape.is_trimmed), bool)
        self.assertEqual(shape.poles.shape[2], 3)
        self.assertEqual(shape.trim_domain.shape, (2, 2))
        self.assertEqual(type(shape.u_closed), bool)
        self.assertEqual(type(shape.u_degree), int)
        self.assertEqual(shape.u_knots.shape[0], 1)
        self.assertEqual(type(shape.u_rational), bool)
        self.assertEqual(type(shape.v_closed), bool)
        self.assertEqual(type(shape.v_degree), int)
        self.assertEqual(shape.v_knots.shape[0], 1)
        self.assertEqual(type(shape.v_rational), bool)


        # sample points
        umin_value, umax_value, vmin_value, vmax_value = shape.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value, 100)
        gridY = np.linspace(vmin_value, vmax_value, 100)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(shape.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        # print(shape.sample(sample_points) - fast.sample(sample_points))
        self.assertTrue(np.allclose(shape.sample(sample_points), fast.sample(sample_points)))

        # derivative

        du, dv, d2u, d2v = surface_derivative(shape, sample_points)
        self.assertTrue(du < 1e-6)
        self.assertTrue(dv < 1e-6)
        self.assertTrue(d2u < 1e-6)
        self.assertTrue(d2v < 1e-6)

        fdu, fdv = fast.first_derivative(sample_points)
        fduu, fdvv, fduv = fast.second_derivative(sample_points)

        # test second deriv
        deriv1 = shape.derivative(sample_points, 1)
        self.assertTrue(np.allclose(deriv1[:,:,0], fdu))
        self.assertTrue(np.allclose(deriv1[:,:,1], fdv))


        deriv2 = shape.derivative(sample_points, 2)
        self.assertTrue(np.allclose(deriv2[:,:,0,0], fduu))
        self.assertTrue(np.allclose(deriv2[:,:,0,1], fduv))
        self.assertTrue(np.allclose(deriv2[:,:,1,0], fduv))
        self.assertTrue(np.allclose(deriv2[:,:,1,1], fdvv))




if __name__ == '__main__':
    unittest.main()
