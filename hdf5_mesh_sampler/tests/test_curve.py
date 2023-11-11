import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
from geometry import Curve
class TestCurve(unittest.TestCase):
    def test_initialization(self):
        curve = Curve()
        self.assertIsNotNone(curve)

    def test_sample(self):
        curve = Curve()
        self.assertEqual(curve.sample([]), [])

if __name__ == '__main__':
    unittest.main()
