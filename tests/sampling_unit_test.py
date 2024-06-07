import unittest

import numpy as np

from test_utilities import *
import abs.sampling as cs



class TestSampling(unittest.TestCase):
    def test_sample_2dcurve(self):
        curve = test_circle2d()
        cs.curve_sampler.uniform_sample(curve, 0.01)
        cs.curve_sampler.random_sample(curve, 0.01)
        cs.curve_sampler.uniform_parametric_sample(curve, 0.01)
        cs.curve_sampler.random_parametric_sample(curve, 0.01)

    def test_sample_3dcurve(self):
        curve = test_ellipse3d()
        cs.curve_sampler.uniform_sample(curve, 0.01)
        cs.curve_sampler.random_sample(curve, 0.01)
        cs.curve_sampler.uniform_parametric_sample(curve, 0.01)
        cs.curve_sampler.random_parametric_sample(curve, 0.01)

    def test_bspline_curve2d(self):
        curve = test_bspline_curve2d()
        cs.curve_sampler.uniform_sample(curve, 0.01)
        cs.curve_sampler.random_sample(curve, 0.01)
        cs.curve_sampler.uniform_parametric_sample(curve, 0.01)
        # cs.curve_sampler.random_parametric_sample(curve, 0.01)



if __name__ == '__main__':
    unittest.main()