"""Main module."""

from shape_sampling import ShapeSampling
from sampling.curve_sampler import CurveSampler
from sampling.surface_sampler import SurfaceSampler
from utilities import *
import os

# For pipeline integration, you can use the following function
def sample_shape(file_path, config):
    """Sample a single shape based on the provided configuration."""

    Data = read_file(file_path)
    data_path_geo = Data.get('geometry/parts')
    data_path_topo = Data.get('topology/parts')

    shape = ShapeSampling(data_path_geo, data_path_topo)
    curve_sampler = CurveSampler(spacing=config['properties']['point_distance']['value'], method="uniform")  # Adjust based on config
    surface_sampler = SurfaceSampler(spacing=config['properties']['point_distance']['value'], method="uniform")  # Adjust based on config

    sampled_shapes = shape.sample_all_shapes(surface_sampler, curve_sampler)
    combined_points = combine_shapes(sampled_shapes)
    downsampled_points = down_sample_point_cloud_pcu(combined_points, target_num_points=config['properties']['target_sample_points']['value']) if config['properties']['use_pcu']['value'] else down_sample_point_cloud(combined_points, target_num_points=config['properties']['target_sample_points']['value'])

    return downsampled_points


# To work locally, you can use the following main function
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
        save_individual_shapes = False
        use_pcu = True
        save_original_sampled_model = True

        # Initialize samplers
        curve_sampler = CurveSampler(spacing=spacing, method="uniform")
        surface_sampler = SurfaceSampler(spacing=spacing, method="uniform")

        # Sample the shape
        sampled_shapes = shape.sample_all_shapes(surface_sampler, curve_sampler)
        combined_points = combine_shapes(sampled_shapes)

        downsampled_points = []
        if use_pcu:
            downsampled_points = down_sample_point_cloud_pcu(combined_points, target_num_points=10000)
        else:
            downsampled_points = down_sample_point_cloud(combined_points, target_num_points=10000)

        # Save the combined and downsampled points
        save_points_to_file(downsampled_points, (input_file_name.split(".")[0] or "shape") + "_downsampled.obj")
        if save_individual_shapes:
            extract_and_save_individual_shapes(sampled_shapes, base_file_name=input_file_name.split(".")[0] or "shape")
        if save_original_sampled_model:
            save_combined_shapes(sampled_shapes, input_file_name.split(".")[0] + "_combined_shapes.obj")


    except Exception as e:
        raise RuntimeError(f"Error in data retrieval or object initialization: {str(e)}")


if __name__ == "__main__":
    main()
