from config_loader import load_config
from common import get_file_paths, split_dataset
from abs import sample_shape
import os
from utilities import save_points_to_file
import argparse


def main(config_path):
    config = load_config(config_path)
    source_path = config['dataset']['data_source']['source_path']
    split_ratios = config['dataset']['split_ratios']

    file_paths = get_file_paths(source_path)
    train_files, validation_files, test_files = split_dataset(file_paths, split_ratios)

    for file_path in train_files:
        try:
            print("Converting Training file: ", file_path)
            base_name = os.path.basename(file_path)
            base_name_no_ext = base_name.rsplit('.', 1)[0]
            new_base_name = f"{base_name_no_ext}-hdf5_downsampled.obj"
            output_path = os.path.join(config['output']['paths']['train_path'], new_base_name)
            downsampled_points = sample_shape(file_path, config)
            save_points_to_file(downsampled_points, output_path)
        except Exception as e:
            print(f"Failed to convert {file_path}. Error: {e}")

    for file_path in validation_files:
        try:
            print("Converting validation file: ", file_path)
            base_name = os.path.basename(file_path)
            base_name_no_ext = base_name.rsplit('.', 1)[0]
            new_base_name = f"{base_name_no_ext}-hdf5_downsampled.obj"
            output_path = os.path.join(config['output']['paths']['validation_path'], new_base_name)
            downsampled_points = sample_shape(file_path, config)
            save_points_to_file(downsampled_points, output_path)
        except Exception as e:
            print(f"Failed to convert {file_path}. Error: {e}")

    for file_path in test_files:
        try:
            print("Converting test file: ", file_path)
            base_name = os.path.basename(file_path)
            base_name_no_ext = base_name.rsplit('.', 1)[0]
            new_base_name = f"{base_name_no_ext}-hdf5_downsampled.obj"
            output_path = os.path.join(config['output']['paths']['test_path'], new_base_name)
            downsampled_points = sample_shape(file_path, config)
            save_points_to_file(downsampled_points, output_path)
        except Exception as e:
            print(f"Failed to convert {file_path}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset')
    parser.add_argument('-c','--config', type=str, help='Path to the config file')
    args = parser.parse_args()
    if args.config:
        main(args.config)
