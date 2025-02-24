import os
import h5py
from abs.utils import *
from abs.part_processor import *
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from abs.shape import *
import numpy as np


def get_normal_func(part, topo, points):
    if isinstance(topo, Face):
        return topo.normal(points)
    else:
        return None


def process_file(file_path, num_samples, get_normal_func):
    """Process a single HDF5 file to generate sample points and normals in a memory-efficient way."""
    with h5py.File(file_path, 'r') as hdf:
        geo = hdf['geometry/parts']
        topo = hdf['topology/parts']

        parts = []
        for i in range(len(geo)):
            # Load data lazily using h5py
            s = Shape(geo.get(list(geo.keys())[i]), topo.get(list(topo.keys())[i]))
            parts.append(s)

        # Generate points and normals using lazy access if possible
        P, S = get_parts(parts, num_samples, get_normal_func)

    return P, S


def save_to_ply(points, normals, file_path):
    """Save the provided points and normals to a .ply file."""
    assert len(points) == len(normals), "Points and normals length must match"

    with open(file_path, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")

        # Write vertex data (points + normals)
        for p, n in zip(points, normals):
            f.write(f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]}\n")


def process_and_save_single_file(file_path, num_samples, get_normal_func, output_dir):

    try:
        points, normals = process_file(file_path, num_samples, get_normal_func)
        for i in range(len(points)):
            output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}_part_{i}.ply")
            save_to_ply(points[i], normals[i], output_file)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise


def process_directory_parallel(directory_path, output_dir, num_samples, get_normal_func, max_workers=4):

    os.makedirs(output_dir, exist_ok=True)
    hdf5_files = [f for f in os.listdir(directory_path) if f.endswith('.hdf5')]
    error_files = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_name in hdf5_files:
            file_path = os.path.join(directory_path, file_name)
            futures.append(
                executor.submit(process_and_save_single_file, file_path, num_samples, get_normal_func, output_dir)
            )

        for future in tqdm(as_completed(futures), total=len(hdf5_files), desc="Processing files"):
            try:
                future.result()
            except Exception as e:
                file_index = futures.index(future)
                failed_file = hdf5_files[file_index]
                error_files.append(failed_file)  # Log the name of the problematic file
                print(f"Task failed for {failed_file}: {e}")

    # Report the files that failed
    if error_files:
        print("\nThe following files failed to process:")
        for error_file in error_files:
            print(error_file)
    else:
        print("\nAll files processed successfully.")


if __name__ == "__main__":


    input_data_file_path = '/Users/nafiseh/Desktop/surface_data/'
    output_dir = '/Users/nafiseh/Desktop/SR4000/'

    num_samples = 4000


    process_directory_parallel(input_data_file_path, output_dir, num_samples, get_normal_func, max_workers=8)
