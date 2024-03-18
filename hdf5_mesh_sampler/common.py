"""The common module contains common functions and classes used by the other modules.
"""
import json

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config
