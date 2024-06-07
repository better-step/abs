"""Main module."""

from shape_sampling import ShapeSampling
from sampling.curve_sampler import CurveSampler
from sampling.surface_sampler import SurfaceSampler
from shape_analysis import ShapeAnalysis
from utilities import *
import os
import json
import pickle

# For pipeline integration, you can use the following function
def sample_shape(file_path, config):
    """Sample a single shape based on the provided configuration."""

    Data = read_file(file_path)
    data_path_geo = Data.get('geometry/parts')
    data_path_topo = Data.get('topology/parts')
    shape = ShapeSampling(data_path_geo, data_path_topo)
    curve_sampler = CurveSampler(spacing=config['dataset']['properties']['point_distance']['value'], method=config['dataset']['properties']['sample_type']['value'])  # Adjust based on config
    surface_sampler = SurfaceSampler(spacing=config['dataset']['properties']['point_distance']['value'], method=config['dataset']['properties']['sample_type']['value'])  # Adjust based on config

    sampled_shapes = shape.sample_all_shapes(surface_sampler, curve_sampler)
    combined_points = combine_shapes(sampled_shapes)
    downsampled_points, indexes = down_sample_point_cloud_pcu(combined_points, target_num_points=config['dataset']['properties']['target_sample_points']['value'])

    if indexes is None:
        print("No points to downsample")
    return downsampled_points


# To work locally, you can use the following main function
def main():
    # TODO: Fix direction/interval in 2d curves
    root_folder = os.getcwd()
    input_file_name = 'Box.hdf5'
    file_path = os.path.join(root_folder, "data", "sample_hdf5", input_file_name)
    try:
        Data = read_file(file_path)
        data_path_geo = Data.get('geometry/parts')
        data_path_topo = Data.get('topology/parts')

        # Initialize the shape with geometry and topology data
        shape_core = ShapeSampling(data_path_geo, data_path_topo)   # TODO: This code only works for shapes with single part

        spacing = 0.4
        save_individual_shapes = False
        save_original_sampled_model = True

        # Initialize samplers
        curve_sampler = CurveSampler(spacing=spacing, method="random")
        surface_sampler = SurfaceSampler(spacing=spacing, method="random")

        # Sample the shape
        sampled_shapes = shape_core.sample_all_shapes(surface_sampler, curve_sampler)

        shape_analysis = ShapeAnalysis(shape_core)

        combined_points, indexes_range = combine_shapes_with_index(sampled_shapes)


        downsampled_points = []
        indexes = []
        downsampled_points, indexes = down_sample_point_cloud_pcu(combined_points, target_num_points=10000)

        sorted_downsampled_points = sorted(list(zip(indexes, downsampled_points)))

        mapped_downsampled_points = map_points_to_ranges(indexes_range, sorted_downsampled_points)

        normals_dict = {}
        for surface_index, surface_points in mapped_downsampled_points.items():
            if isinstance(surface_points, list) and len(surface_points) > 0:
                # normals_dict[f"{part_name}_{shape_name}"] = calculate_normals(shape)
                normals_dict[surface_index] = normals_dict.get(surface_index, [])
                normals_dict[surface_index] = shape_analysis.get_surface_normal(surface_index, surface_points)

        surface_labes = {}
        for surface_index, surface_points in sampled_shapes[0].items():
            if isinstance(surface_points, np.ndarray) and surface_points.size > 0:
                surface_labes[surface_index] = shape_analysis.get_surface_label(surface_index)
        print("")

        # combine downsampled points, normals and labels to one dictionary with nested dictionaries
        output_data = [mapped_downsampled_points, normals_dict, surface_labes]

        # Save as Pickle file
        with open('output_data.pkl', 'wb') as file:
            pickle.dump(output_data, file)

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
