"""The common module contains common functions and classes used by the other modules.
"""

import os

def get_file_paths(source_path):
    """List all HDF5 files in the source directory."""
    return [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.hdf5')]

def split_dataset(file_paths, split_ratios):
    """Split the dataset into train, validation, and test sets based on the given ratios."""
    # Example implementation, adjust as necessary
    total_files = len(file_paths)
    train_end = int(total_files * split_ratios['train'])
    validation_end = train_end + int(total_files * split_ratios['validation'])

    train_files = file_paths[:train_end]
    validation_files = file_paths[train_end:validation_end]
    test_files = file_paths[validation_end:]

    return train_files, validation_files, test_files

