"""Main module."""

from shape_sampling import ShapeSampling
from geometry.curve import Line, Circle, Ellipse, BSplineCurve
from geometry.surface import Plane, Cylinder, Cone, Sphere, Torus, BSplineSurface
from topology.topology import Topology
from sampling.curve_sampler import CurveSampler
from sampling.surface_sampler import SurfaceSampler
import numpy as np
from utilities import save_points,read_file
import os

def main():

    root_folder = os.getcwd()
    input_file_name = 'Cone.hdf5'
    file_path = os.path.join(root_folder, "data", "sample_hdf5", input_file_name)
    try:
        Data = read_file(file_path)
        data_path_geo = Data.get('geometry/parts')
        data_path_topo = Data.get('topology/parts')

        # Initialize the shape with geometry and topology data
        shape = ShapeSampling(data_path_geo, data_path_topo)

        spacing = 1

        # Initialize samplers
        curve_sampler = CurveSampler(spacing=spacing)
        surface_sampler = SurfaceSampler(spacing=spacing)

        # Sample the shape
        shape.sample_all_shapes(surface_sampler, curve_sampler)



        print("Shape initialized")


        print("Data read")
    except Exception as e:
        raise RuntimeError(f"Error in data retrieval or object initialization: {str(e)}")



if __name__ == "__main__":
    main()
