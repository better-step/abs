# import os
# import h5py
# import pickle
# import gzip
# from abs.utils import *
# from abs.part_processor import *
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from joblib import dump
# from abs.shape import *
# from tqdm import tqdm
#
#
# def get_normal_func(part, topo, points):
#
#     if topo.is_face():
#         return topo.normal(points)
#     else:
#         return None
#
# def process_file(file_path, num_samples, get_normal_func):
#
#     with h5py.File(file_path, 'r') as hdf:
#         geo = hdf['geometry/parts']
#         topo = hdf['topology/parts']
#
#         parts = read_parts(file_path)
#
#         P, S = sample_parts(parts, num_samples, get_normal_func)
#
#     return P, S
#
#
# def save_to_pickle(data, file_path):
#     dump(data, file_path)
#
#
# def process_and_save_single_file(file_path, num_samples, get_normal_func, output_dir):
#
#     try:
#         points, normals = process_file(file_path, num_samples, get_normal_func)
#         for i in range(len(points)):
#             part_data = {
#                 'file': os.path.basename(file_path),
#                 'part': i,
#                 'points': points[i],
#                 'normals': normals[i]
#             }
#             output_file = os.path.join(output_dir, f"{os.path.basename(file_path)}_part_{i}.pkl")
#             save_to_pickle(part_data, output_file)
#     except Exception as e:
#         print(f"Error processing file {file_path}: {e}")
#         raise
#
#
#
#
#
# def process_directory_parallel(directory_path, output_dir, num_samples, get_normal_func, max_workers=4):
#
#     os.makedirs(output_dir, exist_ok=True)
#     hdf5_files = [f for f in os.listdir(directory_path) if f.endswith('.hdf5')]
#     error_files = []
#
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         for file_name in hdf5_files:
#             file_path = os.path.join(directory_path, file_name)
#             futures.append(
#                 executor.submit(process_and_save_single_file, file_path, num_samples, get_normal_func, output_dir)
#             )
#
#         for future in tqdm(as_completed(futures), total=len(hdf5_files), desc="Processing files"):
#             try:
#                 future.result()
#             except Exception as e:
#                 file_index = futures.index(future)
#                 failed_file = hdf5_files[file_index]
#                 error_files.append(failed_file)  # Log the name of the problematic file
#                 print(f"Task failed for {failed_file}: {e}")
#
#     if error_files:
#         print("\nThe following files failed to process:")
#         for error_file in error_files:
#             print(error_file)
#     else:
#         print("\nAll files processed successfully.")
#
#
# if __name__ == "__main__":
#
#     input_data_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hdf5')
#     input_data_file_path = os.path.normpath(input_data_file_path)
#     input_data_file_path = '/Users/nafiseh/Desktop/hdf5'
#
#     output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
#     output_dir = os.path.normpath(output_dir)
#
#     num_samples = 2000
#
#     process_directory_parallel(input_data_file_path, output_dir, num_samples, get_normal_func, max_workers=8)
