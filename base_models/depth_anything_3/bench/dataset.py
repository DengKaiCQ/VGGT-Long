# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# References:
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/dataset.py

"""
Base dataset class for benchmark evaluation.

All dataset implementations should inherit from this class and implement
the required abstract methods.
"""

import os
import time
from abc import abstractmethod
from typing import Dict as TDict

import numpy as np
import torch
from addict import Dict

from depth_anything_3.bench.utils import compute_pose
from depth_anything_3.utils.geometry import as_homogeneous


def _wait_for_file_ready(path: str, timeout: float = 3.0, interval: float = 0.2) -> None:
    """Wait until file size stabilizes for 2 consecutive checks."""
    last_size = -1
    stable_count = 0
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(interval)
        size = os.path.getsize(path)
        if size == last_size and size > 0:
            stable_count += 1
            if stable_count >= 2:  # Need 2 consecutive stable checks
                return
        else:
            stable_count = 0
        last_size = size


class Dataset:
    """
    Base class for all benchmark datasets.

    Subclasses must implement:
        - SCENES: List of scene identifiers
        - data_root: Path to dataset root
        - get_data(scene): Return scene data (images, intrinsics, extrinsics, etc.)
        - eval3d(scene, fuse_path): Evaluate 3D reconstruction
        - fuse3d(scene, result_path, fuse_path, mode): Fuse depth maps into point cloud

    Optional overrides:
        - eval_pose(scene, result_path): Evaluate pose estimation (default provided)
    """

    # Subclasses should define these
    SCENES: list = []
    data_root: str = ""

    def __init__(self):
        pass

    def eval_pose(self, scene: str, result_path: str) -> TDict[str, float]:
        """
        Evaluate camera pose estimation accuracy.

        Args:
            scene: Scene identifier
            result_path: Path to .npz file containing predicted extrinsics

        Returns:
            Dict with pose metrics (auc30, auc15, auc05, auc03)
        """
        _wait_for_file_ready(result_path)
        pred = np.load(result_path)
        gt = self.get_data(scene)
        return compute_pose(
            torch.from_numpy(as_homogeneous(pred["extrinsics"])),
            torch.from_numpy(as_homogeneous(gt["extrinsics"])),
        )

    @abstractmethod
    def get_data(self, scene: str) -> Dict:
        """
        Get scene data including images, camera parameters, and auxiliary info.

        Args:
            scene: Scene identifier

        Returns:
            Dict with:
                - image_files: List[str] - paths to images
                - extrinsics: np.ndarray [N, 4, 4] - camera extrinsics (world-to-camera)
                - intrinsics: np.ndarray [N, 3, 3] - camera intrinsics
                - aux: Dict - auxiliary data (masks, GT paths, etc.)
        """
        raise NotImplementedError

    @abstractmethod
    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:
        """
        Evaluate 3D reconstruction quality against ground truth.

        Args:
            scene: Scene identifier
            fuse_path: Path to fused point cloud (.ply)

        Returns:
            Dict with reconstruction metrics (e.g., acc, comp, overall)
        """
        raise NotImplementedError

    @abstractmethod
    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:
        """
        Fuse per-view depth maps into a single point cloud.

        Args:
            scene: Scene identifier
            result_path: Path to .npz file with predicted depths and poses
            fuse_path: Output path for fused point cloud (.ply)
            mode: Fusion mode ("recon_unposed" or "recon_posed")
        """
        raise NotImplementedError

