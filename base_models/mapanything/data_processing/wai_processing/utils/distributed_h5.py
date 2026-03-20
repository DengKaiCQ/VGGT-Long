# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/utils/distributed_h5.py

"""
Distributed H5 Reader from UFM (https://uniflowmatch.github.io/)
Author: Yuchen Zhang (CMU)
"""

import json
import os
from typing import Dict, List, Optional

import h5py
import torch


class DistributedH5Reader:
    def __init__(
        self,
        base_dir: str,
    ):
        """
        Initializes the DistributedH5Reader class based on the specified mode.

        Args:
            base_dir (str): The base directory where HDF5 files will be read from.

            under this directory, all H5 files are assumed to store a part of the dataset,
            and the .json files will be assumed as a dict from dataset index to a tuple
            of (filename, index in the file)
        """

        self.base_dir = base_dir
        self.index_to_file_map = {}
        self._load_and_concatenate_index_maps()
        assert len(self.index_to_file_map) > 0, "No index map found in read mode."

    def __len__(self) -> int:
        """
        Returns the total number of data points in the dataset.

        Returns:
            int: The total number of data points in the dataset.
        """
        return len(self.index_to_file_map)

    def read(
        self, index: int, keys: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Reads a data point from the appropriate HDF5 file based on its index.

        Args:
            index (int): The index of the data point to read.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the data fields and their corresponding tensors.

        Raises:
            KeyError: If the index is not found in the index-to-file map.
        """
        if index in self.index_to_file_map:
            ret = self.index_to_file_map[index]

            if isinstance(ret, list):
                file_name, file_index = ret
            elif isinstance(ret, dict):
                file_name = ret["file_name"]
                file_index = ret["file_index"]

            file_path = os.path.join(self.base_dir, file_name)

            with h5py.File(
                file_path, "r"
            ) as h5file:  # we *could* optimize for repeated file opening here
                read_keys = (
                    [k for k in h5file.keys() if k != "converted_to_rgb"]
                    if keys is None
                    else keys
                )
                try:
                    return {
                        key: torch.from_numpy(h5file[key][file_index])
                        for key in read_keys
                    }
                except KeyError as e:
                    print(
                        f"Error reading keys {read_keys} from file {file_path} at index {file_index}."
                    )
                    raise e
        else:
            raise KeyError(f"Index {index} not found in any file.")

    def _load_and_concatenate_index_maps(self) -> None:
        """
        Loads and concatenates index maps from all divisions in the base directory.

        Args:
            file_name (str): The name of the file from which the index maps will be loaded (default is "index_map.json").
        """
        for file in os.listdir(self.base_dir):
            if file.endswith(".json"):
                file_path = os.path.join(self.base_dir, file)
                with open(file_path, "r") as f:
                    loaded_map = json.load(f)

                if (
                    isinstance(loaded_map, dict)
                    and "forward_match_errors" not in loaded_map
                ):
                    self.index_to_file_map.update(
                        {int(k): v for k, v in loaded_map.items()}
                    )
                else:
                    continue
