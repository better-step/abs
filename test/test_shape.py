from abs.utils import *
from abs.part_processor import *
import unittest
import os


def face_func(face, points):
    return face.normal(points)


def curve_func(edge, points):
    return None

class TestShapeFunctions(unittest.TestCase):

    def test_get_parts_integration(self):
        name = 'Cone'
        sample_name = f'{name}.hdf5'
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_hdf5', sample_name)
        file_path = os.path.normpath(file_path)

        num_samples = 5000
        parts = read_parts(file_path)
        face_points, face_feat, edge_points, edge_feat = sample_parts(parts, num_samples, face_func, curve_func)

        self.assertEqual(len(face_points), len(parts))
        self.assertEqual(len(face_feat), len(parts))
        self.assertEqual(len(face_points), len(face_feat))

if __name__ == '__main__':
    unittest.main()
