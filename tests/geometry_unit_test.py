import unittest

import numpy as np

from Geometry import *
from munch import *


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
    epsilon = 1e-6

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


class TestBasic(unittest.TestCase):

    def test_line3d(self):
        data = {
            "type": "Line",
            "location": [0, 0, 0],
            "direction": [1, 0, 0],
            "interval": [0, 1]
        }

        line = Line(Munch.fromDict(data))
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

    def test_line2d(self):
        data = {
            "type": "Line",
            "location": [0, 0],
            "direction": [1, 0],
            "interval": [0, 1]
        }

        line = Line(Munch.fromDict(data))
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

    def test_circle3d(self):
        data = {
            "type": "Circle",
            "location": [0, 0, 0],
            "radius": 1.0,
            "interval": [0, 2 * np.pi],
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1]
        }

        circle = Circle(Munch.fromDict(data))
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

    def test_circle2d(self):
        data = {
            "type": "Circle",
            "location": [0, 0],
            "radius": 1.0,
            "interval": [0, 2 * np.pi],
            "x_axis": [1, 0],
            "y_axis": [0, 1]
        }

        circle = Circle(Munch.fromDict(data))
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

    def test_ellipse3d(self):
        data = {
            "type": "Ellipse",
            "focus1": [0, 0, 0],
            "focus2": [1, 0, 0],
            "interval": [0, 1],
            "maj_radius": 1.0,
            "min_radius": 0.5,
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1]
        }

        ellipse = Ellipse(Munch.fromDict(data))
        self.assertEqual(ellipse._focus1.shape, (1, 3))
        self.assertEqual(ellipse._focus2.shape, (1, 3))
        self.assertEqual(ellipse._interval.shape, (1, 2))
        self.assertEqual(type(ellipse._maj_radius), float)
        self.assertEqual(type(ellipse._min_radius), float)
        self.assertEqual(ellipse._x_axis.shape, (1, 3))
        self.assertEqual(ellipse._y_axis.shape, (1, 3))
        self.assertEqual(ellipse._z_axis.shape, (1, 3))
        sample_points = np.linspace(0, 2*np.pi, 10).reshape(-1, 1)
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

    def test_ellipse2d(self):
        data = {
            "type": "Ellipse",
            "focus1": [0, 0],
            "focus2": [1, 0],
            "interval": [0, 1],
            "maj_radius": 1.0,
            "min_radius": 0.5,
            "x_axis": [1, 0],
            "y_axis": [0, 1],
            "z_axis": [0, 0]
        }

        ellipse = Ellipse(Munch.fromDict(data))
        self.assertEqual(ellipse._focus1.shape, (1, 2))
        self.assertEqual(ellipse._focus2.shape, (1, 2))
        self.assertEqual(ellipse._interval.shape, (1, 2))
        self.assertEqual(type(ellipse._maj_radius), float)
        self.assertEqual(type(ellipse._min_radius), float)
        self.assertEqual(ellipse._x_axis.shape, (1, 2))
        self.assertEqual(ellipse._y_axis.shape, (1, 2))
        sample_points = np.linspace(0, 2*np.pi, 10).reshape(-1, 1)
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

    def test_plane(self):
        data = {
            "type": "Plane",
            "coefficients": [1, 0, 0, 0],
            "location": [0, 0, 0],
            "trim_domain": [0, 1, 0, 1],
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1]
        }

        plane = Plane(Munch.fromDict(data))
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
        du, dv = surface_derivative(plane, sample_points)
        self.assertTrue(du < 1e-7)
        self.assertTrue(dv < 1e-7)

    def test_cylinder(self):
        data = {
            "type": "Cylinder",
            "location": [0, 0, 0],
            "radius": 1.0,
            "coefficients": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1, 0.0, 0.5, 1],
            "trim_domain": [0, 1, 0, 1],
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1]
        }

        cylinder = Cylinder(Munch.fromDict(data))
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

    def test_cone(self):
        data = {
            "type": "Cone",
            "angle": 0.5,
            "apex": [0, 0, 0],
            "coefficients": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1, 0.0, 0.5, 1],
            "location": [0, 0, 0],
            "radius": 1.0,
            "trim_domain": [0, 1, 0, 1],
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1]
        }

        cone = Cone(Munch.fromDict(data))
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
        data = {
            "type": "Sphere",
            "coefficients": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1, 0.0, 0.5, 1],
            "location": [0, 0, 0],
            "radius": 1.0,
            "trim_domain": [0, 1, 0, 1],
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1]
        }

        sphere = Sphere(Munch.fromDict(data))
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
        data = {
            "type": "Torus",
            "location": [0, 0, 0],
            "max_radius": 1.0,
            "min_radius": 0.5,
            "trim_domain": [0, 1, 0, 1],
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1]
        }

        torus = Torus(Munch.fromDict(data))
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

    def test_bspline_curve2d(self):
        data = {
                    "type": "BSplineCurve",
                    "closed": False,
                    "continuity": 4,
                    "degree": 3,
                    "interval": [3.46944695195361e-18, 0.00133425534020762],
                    "knots": [3.46944695195361e-18, 3.46944695195361e-18, 3.46944695195361e-18, 3.46944695195361e-18,
                            6.0647970009440586e-05, 0.0001212959400188777, 0.0001819439100283148, 0.00024259188003775194,
                              0.00030323985004718907, 0.00036388782005662614, 0.00042453579006606327, 0.0004851837600755004,
                             0.0005458317300849375, 0.0006064797000943747, 0.0006671276701038117, 0.0007277756401132488,
                             0.000788423610122686, 0.0008490715801321231, 0.0009097195501415603, 0.0009703675201509973,
                          0.0010310154901604345, 0.0010916634601698715, 0.0011523114301793087, 0.0012129594001887459,
                             0.0012736073701981828, 0.00133425534020762, 0.00133425534020762, 0.00133425534020762,
                             0.00133425534020762],
                    "poles": [[4.9650692355267685, 17.494250000000005],
                                [4.961032874370447, 17.49425],
                                [4.952977520669924, 17.493791778653545],
                                [4.940944974056901, 17.4917230120382],
                                [4.9289612766339115, 17.488266040182666],
                                [4.917024838625153, 17.483414092751246],
                                [4.905134098009328, 17.477160398261084],
                                [4.893287539849633, 17.46949818950921],
                                [4.881483643961395, 17.460420683320176],
                                [4.8697210821991295, 17.449921156128774],
                                [4.857997988472691, 17.43799266190103],
                                [4.846314687796942, 17.424629084868016],
                                [4.8346635160710045, 17.40982121066919],
                                [4.823066817519954, 17.393571389045626],
                                [4.811553478951728, 17.375897830257383],
                                [4.800122534023212, 17.35680718046304],
                                [4.78878115074683, 17.336309184412688],
                                [4.777534441903705, 17.31441275659142],
                                [4.766388184656843, 17.291127033953135],
                                [4.755348081572766, 17.266461093841347],
                                [4.744419948230315, 17.24042402957224],
                                [4.733609654800948, 17.213024930181962],
                                [4.722923127191439, 17.184272885853893],
                                [4.715885281266602, 17.164208952926344],
                                [4.712388980384689, 17.1539545256125]],
                    "rational": False,
                    "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }

        bspline_curve2d = BSplineCurve(Munch.fromDict(data))
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
        self.assertTrue(d2 < 1e-4)

    def test_bspline_curve3d(self):
        data = {

        }

    def test_bspline_surface(self):

        data = {
            "continuity": 6,
            "face_domain": [0.0, 0.785398163397494, 1.5707963267949, 3.14159265358979],
            "is_trimmed": True,
            "poles":
                [[[-0.498364597137595, 10.0, 5.67148872467595],
                [-0.498364597137595, 10.0, 7.00482205800928],
                [-0.498364597137595, 8.66666666666666, 7.00482205800928]],
                [[-0.22222222222218102, 10.0, 5.67148872467595],
                [-0.22222222222218102, 10.0, 7.00482205800928],
                [-0.22222222222218102, 8.66666666666666, 7.00482205800928]],
                [[-0.026960076346542003, 9.80473785412434, 5.67148872467595],
                [-0.026960076346542003, 9.80473785412434, 6.80955991213362],
                [-0.026960076346542003, 8.66666666666666, 6.80955991213362]]],
            "trim_domain": [0.0, 0.785398163397494, 1.5707963267949, 3.141592653589791],
            "type": "BSpline",
            "u_closed": False,
            "u_degree": 2,
            "u_knots": [0.0, 0.0, 0.0, 0.785398163397494, 0.785398163397494, 0.785398163397494],
            "u_rational": True,
            "v_closed": False,
            "v_degree": 2,
            "v_knots": [1.5707963267949, 1.5707963267949, 1.5707963267949, 3.14159265358979,
                      3.14159265358979, 3.14159265358979],
            "v_rational": True,
            "weights":
                [[1.0, 0.707106781186548, 1.0],
                [0.923879532511278, 0.653281482438182, 0.923879532511278],
                [1.0, 0.707106781186548, 1.0]]
        }

        bspline_surface = BSplineSurface(Munch.fromDict(data))
        umin_value, umax_value, vmin_value, vmax_value = np.array(bspline_surface._trim_domain.reshape(-1, 1))
        gridX = np.linspace(umin_value, umax_value, 2)
        gridY = np.linspace(vmin_value, vmax_value, 2)
        gridX, gridY = np.meshgrid(gridX, gridY)
        u_values = gridX.reshape((np.prod(gridX.shape),))
        v_values = gridY.reshape((np.prod(gridY.shape),))
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(bspline_surface.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))
