# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/launch/launch_utils.py

import importlib.util
import sys
from pathlib import Path


def _escape_scene_names(_scene_names):
    return "[" + ", ".join([f'"{scene_name}"' for scene_name in _scene_names]) + "]"


def import_function_from_path(module_path: Path, function_name: str):
    """
    Import a function from a Python module specified by its file path.

    Args:
        module_path: Path to the Python module file
        function_name: Name of the function to import from the module

    Returns:
        The imported function

    Raises:
        ValueError: If the module or function cannot be loaded
    """
    # Get the directory and module name
    module_dir = str(module_path.parent)
    module_name = module_path.stem

    # Add the module directory to the front of sys.path
    original_path = sys.path.copy()
    sys.path.insert(0, module_dir)

    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise ValueError(
                f"Could not load module '{module_name}' from path: {module_path}"
            )
        if spec.loader is None:
            raise ValueError(f"Could not load loader for module: {module_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get and return the function
        if not hasattr(module, function_name):
            raise ValueError(
                f"Function '{function_name}' not found in module: {module_name}"
            )
        return getattr(module, function_name)
    except Exception as e:
        # Re-raise with more context
        raise ValueError(
            f"Error importing function '{function_name}' from {module_path}: {str(e)}"
        ) from e
    finally:
        # Restore the original sys.path regardless of success or failure
        sys.path = original_path


def parse_string_to_dict(s):
    """
    Convert string representations like '{process_state_not: [metric_alignment, finished]}'
    into a Python dictionary.
    """
    # Remove curly braces
    s = s.strip("{}")

    # Split by colon to get key and value
    parts = s.split(":", 1)
    if len(parts) != 2:
        return {}

    key = parts[0].strip()
    value = parts[1].strip()

    # Handle list values
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]  # Remove brackets
        value_list = [item.strip() for item in value.split(",")]
        return {key: value_list}

    return {key: value}
