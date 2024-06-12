import os
from pathlib import Path
import h5py
import unittest
from abs.geometry import *
import numpy as np
from tests.geometry_unit_test import surface_derivative, curves_derivative, estimate_length


def read_file(file_path):
    assert Path(file_path).exists(), "Please provide valid file path"
    return h5py.File(file_path, 'r')


def get_file(sample_name):
    return os.path.abspath(os.path.join(os.getcwd(), '..', 'abs', 'data', 'sample_hdf5', sample_name))


class Hdf5test(unittest.TestCase):

    def test_geometry_parts(self):
        file_path = get_file()
        print(file_path)
        data = read_file(file_path)
        self.assertIsNotNone(data)
        self.assertIsNotNone(data['geometry']['parts'])

    def test_line2d(self):
        sample_name = 'cylinder_Hole.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['geometry/parts/part_001/2dcurves/001']
            line = Line(grp)
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

        num_samples = 1000  # Can be adjusted for precision
        points = estimate_length(line, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - line.length() < 1e-4))

        lines = np.diff(points, axis=0)
        lengths = np.linalg.norm(lines, axis=1)
        normalized_lines = lines / lengths[:, np.newaxis]
        rotation_matrix = np.array([[0, -1], [1, 0]])
        rotated_p = normalized_lines @ rotation_matrix.T
        self.assertTrue(np.allclose(rotated_p, line.normals(points)))

    def test_circle2d(self):
        sample_name = 'cylinder_Hole.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['geometry/parts/part_001/2dcurves/004']
            circle = Circle(grp)

        sample_points = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        self.assertEqual(circle.sample(sample_points).shape, (10, 2))

        self.assertEqual(circle._location.shape, (1, 2))
        self.assertEqual(type(circle._radius), float)
        self.assertEqual(circle._interval.shape, (1, 2))
        self.assertEqual(circle._x_axis.shape, (1, 2))
        self.assertEqual(circle._y_axis.shape, (1, 2))

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
        num_samples = 1000  # Can be adjusted for precision
        points = estimate_length(circle, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - circle.length() < 1e-4))

        # normals
        lines = np.diff(points, axis=0)
        lengths = np.linalg.norm(lines, axis=1)
        normalized_lines = lines / lengths[:, np.newaxis]
        rotation_matrix = np.array([[0, -1], [1, 0]])
        rotated_p = normalized_lines @ rotation_matrix.T
        self.assertTrue(np.allclose(rotated_p, circle.normals(points)))




    def test_ellipse3d(self):
        sample_name = 'Ellipse.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['geometry/parts/part_001/3dcurves/000']
            ellipse = Ellipse(grp)

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



    def test_bspline_surface(self):
        sample_name = 'Ellipse.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['geometry/parts/part_001/surfaces/000']
            bspline_surface = BSplineSurface(grp)

        self.assertEqual(type(bspline_surface._continuity), int)

    def test_circle3d(self):
        sample_name = 'Cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['geometry/parts/part_001/3dcurves/000']
            circle = Circle(grp)

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

    def test_bspline_curve2d(self):
        sample_name = 'Cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['geometry/parts/part_001/3dcurves/001']
            bspline_curve2d = BSplineCurve(grp)

        # self.assertEqual(bspline_curve2d._closed, False)
        # self.assertEqual(type(bspline_curve2d._continuity), int)
        # self.assertEqual(type(bspline_curve2d._degree), int)
        self.assertEqual(bspline_curve2d._interval.shape, (1, 2))
        self.assertEqual(bspline_curve2d._knots.shape[0], 1)
        # self.assertEqual(bspline_curve2d._poles.shape[1], 2)
        # self.assertEqual(type(bspline_curve2d._rational), bool)
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

        num_samples = 1000  # Can be adjusted for precision
        points = estimate_length(bspline_curve2d, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - bspline_curve2d.length() < 1e-4))

        lines = np.diff(points, axis=0)
        lengths = np.linalg.norm(lines, axis=1)
        normalized_lines = lines / lengths[:, np.newaxis]
        rotation_matrix = np.array([[0, -1], [1, 0]])
        rotated_p = normalized_lines @ rotation_matrix.T
        print('here')

    def test_cone(self):
        sample_name = 'Cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['geometry/parts/part_001/surfaces/000']
            cone = Cone(grp)

        self.assertEqual(cone._apex.shape, (1, 3))
        self.assertEqual(cone._axis.shape, (1, 3))
        self.assertEqual(type(cone._half_angle), float)
        self.assertEqual(type(cone._radius), float)
        self.assertEqual(cone._interval.shape, (1, 2))
        self.assertEqual(cone._x_axis.shape, (1, 3))
        self.assertEqual(cone._y_axis.shape, (1, 3))
        self.assertEqual(cone._z_axis.shape, (1, 3))



