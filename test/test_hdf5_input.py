import os
from pathlib import Path
import h5py
from test_geometry import *
from abs import *
import unittest
import numpy as np
from abs.utils import *


class Hdf5test(unittest.TestCase):

    def test_geometry_parts(self):
        file_path = get_file('cone.hdf5')
        data = h5py.File(file_path, 'r')
        self.assertIsNotNone(data)
        self.assertIsNotNone(data['parts']['part_001']['geometry'])

    #  2D curves
    def test_line2d(self):
        sample_name = 'cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/2dcurves/000']
            line = Line(grp)

        self.assertEqual(line.location.shape, (1, 2))
        self.assertEqual(line.interval.shape, (1, 2))
        self.assertEqual(line.direction.shape, (1, 2))

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
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - line.get_length() < 1e-4))

        # normals
        # rotated_p = estimate_normal(line, num_samples)
        # self.assertTrue(abs(np.sum(rotated_p - line.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        # self.assertTrue(np.allclose(np.linalg.norm(line.normal(param_points), axis=1), 1, atol=1e-8))

    def test_circle2d(self):
        sample_name = 'Cylinder_Hole.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/2dcurves/004']
            circle = Circle(grp)

        sample_points = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        self.assertEqual(circle.sample(sample_points).shape, (10, 2))

        self.assertEqual(circle.location.shape, (1, 2))
        self.assertEqual(type(circle.radius), float)
        self.assertEqual(circle.interval.shape, (1, 2))
        self.assertEqual(circle.x_axis.shape, (1, 2))
        self.assertEqual(circle.y_axis.shape, (1, 2))

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
        param_points, points = generate_points_on_curve(circle, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - circle.get_length() < 1e-4))

        # normals
        # rotated_p = estimate_normal(circle, num_samples)
        # self.assertTrue(abs(np.sum(rotated_p - circle.normal(param_points)[1:, :]) < 1e-4))

        # check if normals are unit length
        # self.assertTrue(np.allclose(np.linalg.norm(circle.normal(param_points), axis=1), 1, atol=1e-8))

    def test_bspline_2dcurve(self):

        sample_name = 'SingleSolidSphere.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/2dcurves/001']
            bspline_curve2d = BSplineCurve(grp)

        self.assertEqual(bspline_curve2d.closed, False)
        self.assertEqual(type(bspline_curve2d.continuity), int)
        self.assertEqual(type(bspline_curve2d.degree), int)
        self.assertEqual(bspline_curve2d.interval.shape, (1, 2))
        self.assertEqual(bspline_curve2d.knots.shape[0], 1)
        self.assertEqual(bspline_curve2d.poles.shape[1], 2)
        self.assertEqual(type(bspline_curve2d.rational), bool)
        self.assertEqual(bspline_curve2d.weights.shape[1], 1)
        self.assertEqual(bspline_curve2d.poles.shape[0], bspline_curve2d.weights.shape[0])

        # sample points
        umin_value, umax_value = bspline_curve2d.interval.T
        gridX = np.linspace(umin_value, umax_value)
        sample_points = gridX.reshape(-1, 1)
        self.assertEqual(bspline_curve2d.sample(sample_points).shape, (gridX.shape[0], 2))

        self.assertEqual(bspline_curve2d.derivative(sample_points, 0).shape, (50, 2))
        self.assertEqual(bspline_curve2d.derivative(sample_points, 1).shape, (50, 2))
        d, d2 = curves_derivative(bspline_curve2d, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)

        # length
        num_samples = 1000  # Can be adjusted for precision
        param_points, points = generate_points_on_curve(bspline_curve2d, num_samples)
        self.assertTrue(abs(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) - bspline_curve2d.get_length() < 1e-4))

    # 3D curves
    def test_line3d(self):
        sample_name = 'Box.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/3dcurves/000']
            line = Line(grp)

        self.assertEqual(line.location.shape, (1, 3))
        self.assertEqual(line.direction.shape, (1, 3))
        self.assertEqual(line.interval.shape, (1, 2))
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
        sample_name = 'cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/3dcurves/000']
            circle = Circle(grp)

        self.assertEqual(circle.location.shape, (1, 3))
        self.assertEqual(type(circle.radius), float)
        self.assertEqual(circle.interval.shape, (1, 2))
        self.assertEqual(circle.x_axis.shape, (1, 3))
        self.assertEqual(circle.y_axis.shape, (1, 3))
        self.assertEqual(circle.z_axis.shape, (1, 3))
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
        sample_name = 'Ellipse.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/3dcurves/000']
            ellipse = Ellipse(grp)

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

    def test_bspline_3dcurve(self):
        sample_name = 'cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/3dcurves/001']
            bspline_curve3d = BSplineCurve(grp)

        self.assertEqual(type(bspline_curve3d.closed), bool)
        self.assertEqual(type(bspline_curve3d.continuity), int)
        self.assertEqual(type(bspline_curve3d.degree), int)
        self.assertEqual(bspline_curve3d.interval.shape, (1, 2))
        self.assertEqual(bspline_curve3d.knots.shape[0], 1)
        self.assertEqual(bspline_curve3d.poles.shape[1], 3)
        self.assertEqual(type(bspline_curve3d.rational), bool)
        self.assertEqual(bspline_curve3d.weights.shape[1], 1)

        # sample points
        umin_value, umax_value = bspline_curve3d.interval.T
        gridX = np.linspace(umin_value, umax_value)
        sample_points = gridX.reshape(-1, 1)
        self.assertEqual(bspline_curve3d.sample(sample_points).shape, (gridX.shape[0], 3))

        # derivative
        self.assertEqual(bspline_curve3d.derivative(sample_points, 0).shape, (50, 3))
        self.assertEqual(bspline_curve3d.derivative(sample_points, 1).shape, (50, 3))
        d, d2 = curves_derivative(bspline_curve3d, sample_points)
        self.assertTrue(d < 1e-4)
        self.assertTrue(d2 < 1e-4)


    # Surfaces
    def test_plane(self):
        sample_name = 'Box.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/surfaces/001']
            plane = Plane(grp)

        self.assertEqual(plane.coefficients.shape, (1, 4))
        self.assertEqual(plane.location.shape, (1, 3))
        self.assertEqual(plane.trim_domain.shape, (2, 2))
        self.assertEqual(plane.x_axis.shape, (1, 3))
        self.assertEqual(plane.y_axis.shape, (1, 3))
        self.assertEqual(plane.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = plane.trim_domain.reshape(-1, 1)
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

    def test_cylinder(self):
        sample_name = 'Cylinder_hole.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/surfaces/000']
            cylinder = Cylinder(grp)

        self.assertEqual(cylinder.location.shape, (1, 3))
        self.assertEqual(type(cylinder.radius), float)
        self.assertEqual(cylinder.coefficients.shape, (1, 10))
        self.assertEqual(cylinder.trim_domain.shape, (2, 2))
        self.assertEqual(cylinder.x_axis.shape, (1, 3))
        self.assertEqual(cylinder.y_axis.shape, (1, 3))
        self.assertEqual(cylinder.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = cylinder.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(cylinder.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        du, dv, d2u, d2v = surface_derivative(cylinder, sample_points)
        self.assertTrue(du < 1e-4)
        self.assertTrue(dv < 1e-4)
        self.assertTrue(d2u < 1e-4)
        self.assertTrue(d2v < 1e-4)


    def test_sphere(self):
        sample_name = 'sphere.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/surfaces/000']
            sphere = Sphere(grp)

        self.assertEqual(sphere.coefficients.shape, (1, 10))
        self.assertEqual(sphere.location.shape, (1, 3))
        self.assertEqual(type(sphere.radius), float)
        self.assertEqual(sphere.trim_domain.shape, (2, 2))
        self.assertEqual(sphere.x_axis.shape, (1, 3))
        self.assertEqual(sphere.y_axis.shape, (1, 3))
        self.assertEqual(sphere.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = sphere.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(sphere.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        # derivative: du, dv
        du, dv, d2u, d2v = surface_derivative(sphere, sample_points)
        self.assertTrue(du < 1e-4)
        self.assertTrue(dv < 1e-4)
        self.assertTrue(d2u < 1e-4)
        self.assertTrue(d2v < 1e-4)

    def test_torus(self):
        sample_name = 'Cylinder_Hole_Fillet_Chamfer.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/surfaces/004']
            torus = Torus(grp)

        self.assertEqual(torus.location.shape, (1, 3))
        self.assertEqual(type(torus.max_radius), float)
        self.assertEqual(type(torus.min_radius), float)
        self.assertEqual(torus.trim_domain.shape, (2, 2))
        self.assertEqual(torus.x_axis.shape, (1, 3))
        self.assertEqual(torus.y_axis.shape, (1, 3))
        self.assertEqual(torus.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = torus.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(torus.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        du, dv, d2u, d2v = surface_derivative(torus, sample_points)
        self.assertTrue(du < 1e-4)
        self.assertTrue(dv < 1e-4)
        self.assertTrue(d2u < 1e-4)
        self.assertTrue(d2v < 1e-4)


    def test_bspline_surface(self):

        sample_name = 'Ellipse.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/surfaces/000']
            bspline_surface = BSplineSurface(grp)

        self.assertEqual(type(bspline_surface.continuity), int)
        self.assertEqual(bspline_surface.face_domain.shape, (1, 4))
        self.assertEqual(type(bspline_surface.is_trimmed), bool)
        self.assertEqual(bspline_surface.poles.shape[2], 3)
        self.assertEqual(bspline_surface.trim_domain.shape, (2, 2))
        self.assertEqual(type(bspline_surface.u_closed), bool)
        self.assertEqual(type(bspline_surface.u_degree), int)
        self.assertEqual(bspline_surface.u_knots.shape[0], 1)
        self.assertEqual(type(bspline_surface.u_rational), bool)
        self.assertEqual(type(bspline_surface.v_closed), bool)
        self.assertEqual(type(bspline_surface.v_degree), int)
        self.assertEqual(bspline_surface.v_knots.shape[0], 1)
        self.assertEqual(type(bspline_surface.v_rational), bool)

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = bspline_surface.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(bspline_surface.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        # derivative
        du, dv, d2u, d2v = surface_derivative(bspline_surface, sample_points, 1e-7)
        self.assertTrue(du < 1e-4)
        self.assertTrue(dv < 1e-4)
        self.assertTrue(d2u < 1e-4)
        self.assertTrue(d2v < 1e-4)

    def test_cone(self):
        sample_name = 'cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/geometry/surfaces/000']
            cone = Cone(grp)

        self.assertEqual(type(cone.angle), float)
        self.assertEqual(cone.apex.shape, (1, 3))
        self.assertEqual(cone.coefficients.shape, (1, 10))
        self.assertEqual(cone.location.shape, (1, 3))
        self.assertEqual(type(cone.radius), float)
        self.assertEqual(cone.trim_domain.shape, (2, 2))
        self.assertEqual(cone.x_axis.shape, (1, 3))
        self.assertEqual(cone.y_axis.shape, (1, 3))
        self.assertEqual(cone.z_axis.shape, (1, 3))

        # sample points
        umin_value, umax_value, vmin_value, vmax_value = cone.trim_domain.reshape(-1, 1)
        gridX = np.linspace(umin_value, umax_value)
        gridY = np.linspace(vmin_value, vmax_value)
        gridX, gridY = np.meshgrid(gridX, gridY)
        sample_points = np.column_stack((gridX, gridY)).reshape(-1, 2)
        self.assertEqual(cone.sample(sample_points).shape, (gridX.shape[0] * gridX.shape[1], 3))

        du, dv, d2u, d2v = surface_derivative(cone, sample_points)
        self.assertTrue(du < 1e-4)
        self.assertTrue(dv < 1e-4)
        self.assertTrue(d2u < 1e-4)
        self.assertTrue(d2v < 1e-4)

if __name__ == '__main__':
    unittest.main()
