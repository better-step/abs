from abs.shape import Shape
from abs.utils import *
from abs.part_processor import *
import unittest


def get_normal_func(shape, geo, points):
    """
    Input:
        - shape
        - geo
        - points: A list or array of points where normal vectors are computed.

    Output:
        - Returns the normal vectors at the provided points as computed by the geo.normal() method.
        - Returns None for curves
    """

    # if geo._shape_name in ['Circle', 'Ellipse'] and len(geo._interval[0]) == 2:
    #     return None
    #
    # if geo._shape_name == 'BSpline':
    #     try:
    #         if len(geo._interval[0]) == 2:
    #             return None
    #     except AttributeError:
    #         pass

    try:
        if len(geo._interval[0]) == 2 or geo._shape_name in 'Other':
            return None
    except AttributeError:
        pass

    normal_points = geo.normal(points)
    return normal_points


def get_labels_func(shape, geo, points):
    """
    Input:
        - shape
        - geo
        - points: A list or array of points where normal vectors are computed.

    Output:
        - Returns the labels at the provided points as computed by the geo.labels() method.
        - Returns None for curves
    """

    labels = geo._shape_name
    return labels


class TestShapeFunctions(unittest.TestCase):

    def test_get_parts_integration(self):
        name = "/Users/nafiseh/Documents/GitHub/abs_new/data/sample_hdf5/Circle"
        sample_name = f'{name}.hdf5'
        num_samples = 1000

        with h5py.File(sample_name, 'r') as hdf:
            geo = hdf['geometry/parts']
            topo = hdf['topology/parts']
            parts = []
            for i in range(len(geo)):
                s = Shape(list(geo.values())[i], list(topo.values())[i])
                parts.append(s)

        # getting the normals:
        P, S = get_parts(parts, num_samples, get_normal_func)

        self.assertEqual(len(P), len(parts))  # P should have the same number of parts as input
        self.assertEqual(len(S), len(parts))  # S should have the same number of parts as input

        self.assertEqual(len(P[0]), len(S[0]))  # P and S should have the same number of samples

        self.assertEqual(len(P[0]), num_samples)  # Each part should have the correct number of samples
        self.assertEqual(len(S[0]), num_samples)  # Each part should have the correct number of samples
