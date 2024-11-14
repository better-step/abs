import os
import h5py
import pickle
import gzip
from abs.shape import Shape
from abs.utils import *
from abs.part_processor import *
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import dump



def get_normal_func(part, topo, points):

    try:
        if hasattr(topo, '_3dcurve'):
            return None
        if topo._surface._shape_name == 'Other':
            return None
    except AttributeError:
        pass

    surf = topo._surface
    surf_orientation = topo._surface_orientation
    shell_orienttion = part.Solid.shells[0]._orientation_wrt_solid
    normal_points = surf.normal(points)

    if not surf_orientation and shell_orienttion:
        normal_points = -normal_points

    # normalize the 3d points to enable it for surface reconstruction


    return normal_points

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


def save_to_pickle(data, file_path):
    """Save the provided data to a compressed pickle file using joblib."""
    dump(data, file_path)


def process_and_save_single_file(file_path, num_samples, get_normal_func, output_dir):
    """Process a single file and save its results to a pickle file."""
    try:
        points, normals = process_file(file_path, num_samples, get_normal_func)
        for i in range(len(points)):
            part_data = {
                'file': os.path.basename(file_path),
                'part': i,
                'points': points[i],
                'normals': normals[i]
            }
            output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}_part_{i}.pkl")
            save_to_pickle(part_data, output_file)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise


def process_directory_parallel(directory_path, output_dir, num_samples, get_normal_func, max_workers=4):
    """Process all HDF5 files in parallel using ProcessPoolExecutor."""
    os.makedirs(output_dir, exist_ok=True)
    hdf5_files = [f for f in os.listdir(directory_path) if f.endswith('.hdf5')]

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
                print(f"Task failed: {e}")



if __name__ == "__main__":
    # Set directory paths and parameters
    input_data_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hdf5')
    input_data_file_path = os.path.normpath(input_data_file_path)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    output_dir = os.path.normpath(output_dir)

    num_samples = 2000  # Define how many samples to generate per part

    # Assuming get_normal_func is already defined and Shape, get_parts are defined properly
    # Process files in parallel
    process_directory_parallel(input_data_file_path, output_dir, num_samples, get_normal_func, max_workers=8)
