from abs.part_processor import *
from abs.utils import *
import unittest
from test_utilities import *


def get_normal_func(part, topo, points):
    if topo.is_face():
        return topo.normal(points)
    else:
        return None


class TestApplication(unittest.TestCase):
    def test_run_application(self):
        file_path = get_file("Ellipse.hdf5")
        num_samples = 4000

        # print('getting parts')

        parts = read_parts(file_path)
        meshes = read_meshes(file_path)


        # print('sampling parts with sample size:', num_samples)
        P, S = sample_parts(parts, num_samples, get_normal_func)
        V, F = get_mesh(meshes)

        # print('saving results')
        # save_file_path = os.path.join(base_dir, f"{file_name}.obj")
        # save_file_path_ply = os.path.join(base_dir, f"{file_name}.ply")

        # save_ply(save_file_path_ply, P, S)
        # save_obj_mesh(save_file_path, V, F)

        # print('finished')