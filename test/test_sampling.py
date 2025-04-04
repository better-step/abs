import unittest
import numpy as np
from test.utils.test_utilities import *
import abs.sampler as cs
from abs import poisson_disk_downsample
import abs.utils as au


class TestSampling(unittest.TestCase):
    def test_sample_2dcurve(self):
        curve = circle2d()
        cs._uniform_sample_curve(curve, 2000)
        cs._random_sample_curve(curve, 2000)

    def test_sample_3dcurve(self):
        curve = ellipse3d()
        cs._uniform_sample_curve(curve, 2000)
        cs._random_sample_curve(curve, 2000)

    def test_sample_bspline_curve2d(self):
        curve = bspline_curve2d()
        cs._uniform_sample_curve(curve, 2000)
        cs._random_sample_curve(curve, 2000)

    def test_downsample(self):
        pts = np.random.rand(1000, 3)
        indices = poisson_disk_downsample(pts, 100)

    #TODO: Fix this test
    # def test_downsample1(self):
    #     pp = plane()
    #     _, pts = cs._random_sample_curve(pp, 5000)
    #     indices = poisson_disk_downsample(pts, 1000, 50)
    #     au.save_obj('test_sampling.obj', pts[indices])


if __name__ == '__main__':
    unittest.main()
