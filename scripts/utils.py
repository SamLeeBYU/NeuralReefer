import os
import sys
import numpy as np

from time import perf_counter  # Runtime measurement
from contextlib import contextmanager

import warnings           # Suppress non-critical runtime warnings
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")

#For verbose functionality
def suppress_prints():
    """Redirects stdout to null to suppress print statements."""
    sys.stdout = open(os.devnull, 'w')

def restore_prints():
    """Restores normal stdout printing."""
    sys.stdout = sys.__stdout__

#The following method was written by ChatGPT 4o
#This helps convert python dictionaries to json-compatible objects
def convert_json_compat(obj):
    """
    Recursively converts numpy datatypes to native Python types
    so that they can be safely serialized to JSON.
    """
    if isinstance(obj, dict):
        return {k: convert_json_compat(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_json_compat(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_json_compat(v) for v in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

@contextmanager
def timer(verbose=True):
    start = perf_counter()
    try:
        yield
    finally:
        if verbose:
            print(f"Time: {perf_counter() - start:2f}s")