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
    # TODO: Fix direction/interval in 2d curves
    root_folder = os.getcwd()
    input_file_name = '20230_21cfb69e_0007_1.hdf5'
    file_path = os.path.join(root_folder, "data", "hdf5", input_file_name)
    try:
        Data = read_file(file_path)
        data_path_geo = Data.get('geometry/parts')
        data_path_topo = Data.get('topology/parts')

        # Initialize the shape with geometry and topology data
        shape = ShapeSampling(data_path_geo, data_path_topo)   # TODO: Addd some class like part geometry for each part on top

        spacing = 0.4

        # Initialize samplers
        curve_sampler = CurveSampler(spacing=spacing, method="uniform")
        surface_sampler = SurfaceSampler(spacing=spacing, method="uniform")

        # Sample the shape
        sampled_shapes = shape.sample_all_shapes(surface_sampler, curve_sampler)



        for part_index, part in sampled_shapes.items():
            print("")
            print("Part: {}".format(part_index))

            for shape_id, shape_points in part.items():
                print("Shape: {}".format(shape_id))
                file_name = f"shape_{shape_id}.obj"
                if shape_points is None or shape_points.size == 0:
                    print(f"Warning: No points to save in {file_name}.")
                    continue

                print(f"Saving shape {shape_id} to {file_name}")
                save_points(shape_points, file_name)

        save_combined_shapes(sampled_shapes, "combined_shapes.obj")

        print("Shape initialized")

    except Exception as e:
        raise RuntimeError(f"Error in data retrieval or object initialization: {str(e)}")


if __name__ == "__main__":
    main()
