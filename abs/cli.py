"""Command-line interface for converting ABS geometry to PLY or Pickle formats."""
from __future__ import annotations

import os
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from joblib import dump
import traceback
from abs.utils import read_parts
from abs.part_processor import sample_parts

def get_normal_func(part, topo, points):
    """Return normal vectors for faces, None for other topology elements."""
    if topo.is_face():
        return topo.normal(points)
    else:
        return None

def to_ply_main():
    """Console script entry point for converting to PLY."""
    parser = argparse.ArgumentParser(prog="abs-to-ply",
                                     description="Sample points and normals from ABS geometry and output as PLY files.")
    parser.add_argument("input_path", help="Path to an input .hdf5 file or a directory containing .hdf5 files")
    parser.add_argument("output_dir", help="Directory to save output PLY files")
    parser.add_argument("-n", "--num-samples", dest="num_samples", type=int, default=2000,
                        help="Number of sample points per part (default: 2000)")
    parser.add_argument("-j", "--jobs", dest="max_workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()
    input_path = args.input_path
    output_dir = args.output_dir
    num_samples = args.num_samples
    max_workers = args.max_workers
    os.makedirs(output_dir, exist_ok=True)
    # Determine if input is a single file or directory of files
    files_to_process = []
    if os.path.isdir(input_path):
        # Process all .hdf5 files in directory
        files_to_process = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.hdf5')]
    elif os.path.isfile(input_path):
        files_to_process = [input_path]
    else:
        parser.error(f"Input path {input_path} does not exist.")
    error_files = []
    # Define worker function
    def process_and_save_ply(file_path):
        try:
            parts = read_parts(file_path)
            P, S = sample_parts(parts, num_samples, get_normal_func)
            base_name = os.path.basename(file_path)
            for i in range(len(P)):
                points = P[i]
                normals = S[i]
                if len(points) != len(normals):
                    raise RuntimeError(f"Points/normals length mismatch for part {i} in {file_path}")
                out_file = os.path.join(output_dir, f"{base_name}_part_{i}.ply")
                with open(out_file, 'w') as f:
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
                    # Write each vertex and normal
                    for p, n in zip(points, normals):
                        f.write(f"{p[0]} {p[1]} {p[2]} {n[0]} {n[1]} {n[2]}\n")
        except Exception as e:
            # Print traceback for debugging
            traceback.print_exc()
            raise
    # Process files in parallel
    if max_workers > 1 and len(files_to_process) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_and_save_ply, fp): fp for fp in files_to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    error_files.append(os.path.basename(file_path))
                    print(f"Error processing file {file_path}: {exc}")
    else:
        # Single-threaded processing
        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                process_and_save_ply(file_path)
            except Exception as exc:
                error_files.append(os.path.basename(file_path))
                print(f"Error processing file {file_path}: {exc}")
    if error_files:
        print("\nThe following files failed to process:")
        for fname in error_files:
            print(fname)
    else:
        print("\nAll files processed successfully.")

def to_pickle_main():
    """Console script entry point for converting to Pickle."""
    parser = argparse.ArgumentParser(prog="abs-to-pickle",
                                     description="Sample points and normals from ABS geometry and output as Pickle files.")
    parser.add_argument("input_path", help="Path to an input .hdf5 file or a directory containing .hdf5 files")
    parser.add_argument("output_dir", help="Directory to save output .pkl files")
    parser.add_argument("-n", "--num-samples", dest="num_samples", type=int, default=2000,
                        help="Number of sample points per part (default: 2000)")
    parser.add_argument("-j", "--jobs", dest="max_workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()
    input_path = args.input_path
    output_dir = args.output_dir
    num_samples = args.num_samples
    max_workers = args.max_workers
    os.makedirs(output_dir, exist_ok=True)
    files_to_process = []
    if os.path.isdir(input_path):
        files_to_process = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.hdf5')]
    elif os.path.isfile(input_path):
        files_to_process = [input_path]
    else:
        parser.error(f"Input path {input_path} does not exist.")
    error_files = []
    def process_and_save_pickle(file_path):
        try:
            parts = read_parts(file_path)
            P, S = sample_parts(parts, num_samples, get_normal_func)
            base_name = os.path.basename(file_path)
            for i in range(len(P)):
                data = {
                    'file': base_name,
                    'part': i,
                    'points': P[i],
                    'normals': S[i]
                }
                out_file = os.path.join(output_dir, f"{base_name}_part_{i}.pkl")
                dump(data, out_file)
        except Exception as e:
            traceback.print_exc()
            raise
    if max_workers > 1 and len(files_to_process) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_and_save_pickle, fp): fp for fp in files_to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                file_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    error_files.append(os.path.basename(file_path))
                    print(f"Error processing file {file_path}: {exc}")
    else:
        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                process_and_save_pickle(file_path)
            except Exception as exc:
                error_files.append(os.path.basename(file_path))
                print(f"Error processing file {file_path}: {exc}")
    if error_files:
        print("\nThe following files failed to process:")
        for fname in error_files:
            print(fname)
    else:
        print("\nAll files processed successfully.")
