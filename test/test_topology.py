import unittest
from pathlib import Path
import h5py
import os
from abs.topology import Topology

from abs.utils import *


class TestTopology(unittest.TestCase):

    def test_topology(self):
        sample_name = 'Cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['topology/parts/part_001/']
            topology = Topology(grp)

    def test_find_adjacent_faces(self):
        sample_name = 'Cylinder_Hole_Fillet.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['topology/parts/part_001/']
            topology = Topology(grp)

        face_index = 2
        adjacent_faces = topology.find_adjacent_faces(face_index)
        print(adjacent_faces)

    def test_find_connected_components(self):
        sample_name = 'Cylinder_Hole_Fillet.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['topology/parts/part_001/']
            topology = Topology(grp)

        connected_components = topology.find_connected_components()
        self.assertEqual(len(connected_components[0]), 7)
