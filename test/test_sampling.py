import unittest

import numpy as np

from utils.test_utilities import *
import abs.sampling as cs
from abs import poisson_disk_downsample

import abs.utils as au


class TestSampling(unittest.TestCase):
    def test_sample_2dcurve(self):
        curve = circle2d()
        cs.curve_sampler.uniform_sample(curve, 0.01)
        cs.curve_sampler.random_sample(curve, 0.01)
        cs.curve_sampler.uniform_parametric_sample(curve, 0.01)
        cs.curve_sampler.random_parametric_sample(curve, 0.01)

    def test_sample_3dcurve(self):
        curve = ellipse3d()
        cs.curve_sampler.uniform_sample(curve, 0.01)
        cs.curve_sampler.random_sample(curve, 0.01)
        cs.curve_sampler.uniform_parametric_sample(curve, 0.01)
        cs.curve_sampler.random_parametric_sample(curve, 0.01)

    def test_sample_bspline_curve2d(self):
        curve = bspline_curve2d()
        cs.curve_sampler.uniform_sample(curve, 0.01)
        cs.curve_sampler.random_sample(curve, 0.01)
        cs.curve_sampler.uniform_parametric_sample(curve, 0.01)
        # cs.curve_sampler.random_parametric_sample(curve, 0.01)

    def test_downsample(self):
        pts = np.random.rand(1000, 3)
        indices = poisson_disk_downsample(pts, 100)

    def test_downsample1(self):
        pp = plane()
        _, pts = cs.surface_sampler.random_sample(pp, 0.005)
        indices = poisson_disk_downsample(pts, 1000, 50)
        au.save_obj('test_sampling.obj', pts[indices])


if __name__ == '__main__':
    unittest.main()