from abs.part_processor import *
from abs.utils import *
import unittest
from test_utilities import *


def face_func(face, points):
    return face.normal(points)


class TestApplication(unittest.TestCase):
    def test_run_application(self):
        file_path = get_file("Ellipse.hdf5")
        num_samples = 4000
        parts = read_parts(file_path)
        meshes = read_meshes(file_path)

        P, S = sample_parts(parts, num_samples, face_func)
        V, F = get_mesh(meshes)

