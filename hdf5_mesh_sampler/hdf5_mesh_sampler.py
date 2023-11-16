"""Main module."""

from shape_sampling import ShapeSampling
from geometry.curve import Line, Circle, Ellipse, BSplineCurve
from geometry.surface import Plane, Cylinder, Cone, Sphere, Torus, BSplineSurface
from topology.topology import Topology
from sampling.curve_sampler import CurveSampler
from sampling.surface_sampler import SurfaceSampler
import numpy as np
from utilities import save_points,read_file, save_combined_shapes
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

        spacing = 0.1

        # Initialize samplers
        curve_sampler = CurveSampler(spacing=spacing, method="uniform")
        surface_sampler = SurfaceSampler(spacing=spacing, method="uniform")

        # Sample the shape
        sampled_shapes = shape.sample_all_shapes(surface_sampler, curve_sampler)

        for shape_id, shape_points in sampled_shapes.items():
            file_name = f"shape_{shape_id}.obj"
            save_points(shape_points, file_name)

        save_combined_shapes(sampled_shapes, "combined_shapes.obj")

        print("Shape sampled")








        print("Shape initialized")

    except Exception as e:
        raise RuntimeError(f"Error in data retrieval or object initialization: {str(e)}")



if __name__ == "__main__":
    main()
