# example of a dataset for normal estimation

import os
import h5py
import pickle
from abs.shape import Shape
from abs.utils import *
from abs.part_processor import *
from tqdm import tqdm


def get_normal_func(shape, geo, points):

    try:
        if len(geo._interval[0]) == 2:
            return None
    except AttributeError:
        pass

    normal_points = geo.normal(points)
    return normal_points

def process_file(file_path, num_samples, get_normal_func):

    """Process a single HDF5 file, generate sample points and normals, and return them."""
    with h5py.File(file_path, 'r') as hdf:
        geo = hdf['geometry/parts']
        topo = hdf['topology/parts']
        parts = []
        for i in range(len(geo)):
            s = Shape(list(geo.values())[i], list(topo.values())[i])
            parts.append(s)

    # Get points and normals
    P, S = get_parts(parts, num_samples, get_normal_func)

    return P, S


def save_to_pickle(data, file_path):
    """Save the provided data to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def process_directory(directory_path, output_pickle_file, num_samples, get_normal_func):
    """Process all HDF5 files in the given directory and save their processed data as a single pickle file."""

    # Initialize an empty list to accumulate data from all files
    all_data = []

    # Get list of all HDF5 files in the directory
    hdf5_files = [f for f in os.listdir(directory_path) if f.endswith('.hdf5')]

    # Use tqdm to create a progress bar
    for file_name in tqdm(hdf5_files, desc="Processing files"):
        file_path = os.path.join(directory_path, file_name)

        print(f"Processing file {file_name}")

        # Process the file to get points and normals
        points, normals = process_file(file_path, num_samples, get_normal_func)

        # Append data for each part in the file to all_data
        for i in range(len(points)):
            part_data = {
                'file': file_name,
                'part': i,
                'points': points[i],
                'normals': normals[i]
            }
            all_data.append(part_data)

    # After processing all files, save all data as a single pickle file
    save_to_pickle(all_data, output_pickle_file)
    print(f"Saved all processed data to {output_pickle_file}")


# Example usage:
directory_path = '/Users/nafiseh/Documents/GitHub/meshInYaml/meshInYaml/data_hdf5/j1.0.0/hdf5'
output_dir = '/Users/nafiseh/Documents/GitHub/abs_new/processed_data/new_train.pkl'
num_samples = 1000

# Assuming get_normal_func is already defined
process_directory(directory_path, output_dir, num_samples, get_normal_func)
