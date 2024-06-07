import json
import os
import sys


def load_config(file_path):
    """Loads and validates the configuration file, and ensures output directories exist."""
    with open(file_path, 'r') as file:
        config = json.load(file)

    if not validate_config(config):
        sys.exit("Configuration validation failed.")

    ensure_output_dirs_exist(config["output"]["paths"])

    return config


def validate_config(config):
    """Validates the loaded configuration."""
    # Validate experimentDetails
    if "experimentDetails" not in config or not validate_experiment_details(config["experimentDetails"]):
        return False

    # Validate dataset
    if "dataset" not in config or not validate_dataset(config["dataset"]):
        return False

    # Validate output paths
    if "output" not in config or not validate_output(config["output"]):
        return False

    return True


def validate_experiment_details(details):
    required_keys = ["name", "description", "tags"]
    return all(key in details for key in required_keys)


def validate_dataset(dataset):
    if "data_source" not in dataset or "source_path" not in dataset["data_source"]:
        print("Missing data source path.")
        return False
    if not os.path.isdir(dataset["data_source"]["source_path"]):
        print("Data source path is not a directory.")
        return False
    if "properties" not in dataset:
        print("Missing dataset properties.")
        return False
    if "split_ratios" not in dataset or not validate_split_ratios(dataset["split_ratios"]):
        return False
    return True


def validate_split_ratios(ratios):
    expected_keys = ["train", "validation", "test"]
    if not all(key in ratios for key in expected_keys):
        print("Missing split ratio.")
        return False
    total_ratio = sum(ratios[key] for key in expected_keys)
    return abs(total_ratio - 1.0) < 0.01  # Allow small margin for floating-point errors


def validate_output(output):
    paths = output.get("paths", {})
    required_keys = ["train_path", "validation_path", "test_path"]
    for key in required_keys:
        if key not in paths or not isinstance(paths[key], str):
            print(f"Missing or invalid output path: {key}")
            return False
    return True


def ensure_output_dirs_exist(paths):
    """Ensures that the specified output directories exist, creating them if necessary."""
    # Filter out the 'description' key or any non-path value
    valid_paths = {k: v for k, v in paths.items() if k.endswith("_path")}

    for path in valid_paths.values():
        if path and not os.path.exists(path):
            try:
                os.makedirs(path)
                print(f"Created directory: {path}")
            except OSError as e:
                sys.exit(f"Failed to create directory {path}: {e}")


# Example usage
if __name__ == "__main__":
    config_path = '/Users/chandu/Workspace/GM/HDF5MeshSampler/abs/config/config.json'
    config = load_config(config_path)
    print("Configuration loaded, validated, and necessary directories ensured.")
