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
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/datasets/dtu64.py

"""
DTU-64 Dataset implementation for POSE EVALUATION ONLY.

This is a subset of DTU with 64 images per scene, specifically designed for
camera pose estimation evaluation. It does NOT support 3D reconstruction.

Note: GT depth loading is not implemented as it's not needed for pose evaluation.
"""

import glob
import os
from typing import Dict as TDict

import numpy as np
from addict import Dict

from depth_anything_3.bench.dataset import Dataset
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.utils.constants import (
    DTU64_CAMERA_ROOT,
    DTU64_EVAL_DATA_ROOT,
    DTU64_SCENES,
)


@MV_REGISTRY.register(name="dtu64")
@MONO_REGISTRY.register(name="dtu64")
class DTU64(Dataset):
    """
    DTU-64 Dataset wrapper for DepthAnything3 POSE EVALUATION ONLY.

    This dataset is a subset of DTU with 64 images per scene.
    It is specifically designed for camera pose estimation evaluation
    and does NOT support 3D reconstruction evaluation.

    Dataset structure:
        DTU/scans/
        ├── {scene}/
        │   └── image/          # RGB images (64 per scene)
        └── Cameras/
            └── {idx}_cam.txt   # Camera parameters

    Supported modes:
        - pose: Camera pose estimation evaluation

    NOT supported:
        - recon_unposed: 3D reconstruction (no GT depth available)
        - recon_posed: 3D reconstruction (no GT depth available)
    """

    data_root = DTU64_EVAL_DATA_ROOT
    camera_root = DTU64_CAMERA_ROOT
    SCENES = DTU64_SCENES

    def __init__(self):
        super().__init__()
        self._scene_cache = {}

    # ------------------------------
    # Camera file parsing
    # ------------------------------

    def read_cam_file(self, filename: str) -> tuple:
        """
        Read DTU camera file containing extrinsics and intrinsics.

        Args:
            filename: Path to camera text file

        Returns:
            Tuple of (intrinsics [3,3], extrinsics [4,4])
        """
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
        return intrinsics, extrinsics

    # ------------------------------
    # Public API
    # ------------------------------

    def get_data(self, scene: str) -> Dict:
        """
        Collect per-view image paths, intrinsics/extrinsics for a scene.

        Args:
            scene: Scene identifier (e.g., "scan105")

        Returns:
            Dict with:
                - image_files: List[str] - paths to images (64 per scene)
                - extrinsics: np.ndarray [N, 4, 4] - world-to-camera transforms
                - intrinsics: np.ndarray [N, 3, 3] - camera intrinsics
                - aux: Dict (empty for this dataset)
        """
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        rgb_folder = os.path.join(self.data_root, scene, "image")

        # Get all PNG files sorted
        files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))

        # Reorder: place index 33 first (reference view convention)
        if len(files) > 33:
            files = [files[33]] + files[:33] + files[34:]

        out = Dict({
            "image_files": [],
            "extrinsics": [],
            "intrinsics": [],
            "aux": Dict({}),
        })

        for rgb_file in files:
            basename = os.path.basename(rgb_file)
            # File naming: "00000033.png" -> cam_idx = 33
            file_idx = basename.split(".")[0]
            cam_idx = int(file_idx)

            # Camera file path
            cam_file = os.path.join(self.camera_root, f"{cam_idx:0>8}_cam.txt")

            if not os.path.exists(cam_file):
                print(f"[DTU-64] Warning: Camera file not found: {cam_file}")
                continue

            intrinsics, extrinsics = self.read_cam_file(cam_file)

            out.image_files.append(rgb_file)
            out.extrinsics.append(extrinsics)
            out.intrinsics.append(intrinsics)

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)

        print(f"[DTU-64] {scene}: {len(out.image_files)} images (pose evaluation only)")

        self._scene_cache[scene] = out
        return out

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:
        """
        NOT SUPPORTED for DTU-64.

        DTU-64 is only for pose evaluation, not 3D reconstruction.
        """
        raise NotImplementedError(
            "DTU-64 dataset is for POSE EVALUATION ONLY. "
            "3D reconstruction evaluation is not supported. "
            "Use the standard 'dtu' dataset for 3D reconstruction evaluation."
        )

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:
        """
        NOT SUPPORTED for DTU-64.

        DTU-64 is only for pose evaluation, not 3D reconstruction.
        """
        raise NotImplementedError(
            "DTU-64 dataset is for POSE EVALUATION ONLY. "
            "3D reconstruction (fuse3d) is not supported. "
            "Use the standard 'dtu' dataset for 3D reconstruction."
        )

