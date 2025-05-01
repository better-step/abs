import unittest
from pathlib import Path
import h5py
import os
from abs.topology import Topology
from abs.utils import *


class TestTopology(unittest.TestCase):

    def test_topology(self):
        sample_name = 'cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/topology']
            geo = hdf['parts/part_001/geometry']
            s = Shape(geo, grp)
            topo_solid = s.Solid


    def test_find_adjacent_faces(self):
        sample_name = 'cone.hdf5'
        file_path = get_file(sample_name)
        with h5py.File(file_path, 'r') as hdf:
            grp = hdf['parts/part_001/topology']
            geo = hdf['parts/part_001/geometry']
            s = Shape(geo, grp)

        face_index = 1
        # adjacent_faces = s.Solid.faces[face_index]
        current_face = s.Solid.faces[face_index]
        adjacent_faces = current_face.find_adjacent_faces()
        print(adjacent_faces)

    # def test_find_connected_components(self):
    #     sample_name = 'cone.hdf5'
    #     file_path = get_file(sample_name)
    #     with h5py.File(file_path, 'r') as hdf:
    #         grp = hdf['parts/part_001/topology']
    #         geo = hdf['parts/part_001/geometry']
    #         s = Shape(geo, grp)
    #
    #     connected_components = topology.find_connected_components()
    #     self.assertEqual(len(connected_components[0]), 7)


    # def find_connected_components(self):
    #     """
    #     Identify connected components in the topology.
    #     """
    #     visited = set()
    #     components = []
    #
    #     def dfs(face_index):
    #         if face_index in visited:
    #             return
    #         visited.add(face_index)
    #         component.add(face_index)
    #         for adjacent_face in self.find_adjacent_faces(face_index):
    #             dfs(adjacent_face)
    #
    #     for face_index in range(len(self.faces)):
    #         if face_index not in visited:
    #             component = set()
    #             dfs(face_index)
    #             components.append(component)
    #
    #     return [list(component) for component in components]
if __name__ == '__main__':
    unittest.main()
