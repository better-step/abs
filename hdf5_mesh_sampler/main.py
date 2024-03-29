from config_loader import load_config
from common import get_file_paths, split_dataset
from .hdf5_mesh_sampler import sample_shape
import os
from utilities import save_points_to_file


def main(config_path):
    config = load_config(config_path)
    source_path = config['dataset']['data_source']['source_path']
    split_ratios = config['dataset']['split_ratios']

    file_paths = get_file_paths(source_path)
    train_files, validation_files, test_files = split_dataset(file_paths, split_ratios)

    for file_path in train_files:
        output_path = os.path.join(config['output']['paths']['train_path'], os.path.basename(file_path))
        downsampled_points = sample_shape(file_path, config)
        save_points_to_file(downsampled_points, output_path + "_downsampled.obj")

    for file_path in validation_files:
        output_path = os.path.join(config['output']['paths']['validation_path'], os.path.basename(file_path))
        downsampled_points = sample_shape(file_path, config)
        save_points_to_file(downsampled_points, output_path + "_downsampled.obj")

    for file_path in test_files:
        output_path = os.path.join(config['output']['paths']['test_path'], os.path.basename(file_path))
        downsampled_points = sample_shape(file_path, config)
        save_points_to_file(downsampled_points, output_path + "_downsampled.obj")


if __name__ == "__main__":
    main("/Users/chandu/Workspace/GM/HDF5MeshSampler/hdf5_mesh_sampler/config/config.json")
