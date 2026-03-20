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
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/datasets/sevenscenes.py

"""
7Scenes Benchmark dataset implementation.

7Scenes is an indoor RGB-D dataset with ground truth camera poses and 3D meshes.
Reference: https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/

Evaluation metrics:
- 3D reconstruction: Accuracy, Completeness, F-score
- Camera pose estimation: AUC metrics
"""

import os
from typing import Dict as TDict

import cv2
import numpy as np
import open3d as o3d
from addict import Dict

from depth_anything_3.bench.dataset import Dataset, _wait_for_file_ready
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    evaluate_3d_reconstruction,
    fuse_depth_to_tsdf,
    sample_points_from_mesh,
)
from depth_anything_3.utils.constants import (
    SEVENSCENES_CX,
    SEVENSCENES_CY,
    SEVENSCENES_DOWN_SAMPLE,
    SEVENSCENES_EVAL_DATA_ROOT,
    SEVENSCENES_EVAL_THRESHOLD,
    SEVENSCENES_FX,
    SEVENSCENES_FY,
    SEVENSCENES_MAX_DEPTH,
    SEVENSCENES_SAMPLING_NUMBER,
    SEVENSCENES_SCENES,
    SEVENSCENES_SDF_TRUNC,
    SEVENSCENES_VOXEL_LENGTH,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama


@MV_REGISTRY.register(name="7scenes")
@MONO_REGISTRY.register(name="7scenes")
class SevenScenes(Dataset):
    """
    7Scenes Benchmark dataset wrapper for DepthAnything3 evaluation.

    Supports:
        - Camera pose estimation evaluation (AUC metrics)
        - 3D reconstruction evaluation (Accuracy, Completeness, F-score)
        - TSDF-based point cloud fusion

    Dataset structure:
        7scenes/
        ├── 7Scenes/
        │   ├── {scene}/
        │   │   └── seq-01/  (or seq-02 for stairs)
        │   │       ├── frame-XXXXXX.color.png
        │   │       ├── frame-XXXXXX.depth.png
        │   │       └── frame-XXXXXX.pose.txt
        │   └── meshes/
        │       └── {scene}.ply  # Ground truth mesh
    """

    data_root = SEVENSCENES_EVAL_DATA_ROOT
    SCENES = SEVENSCENES_SCENES

    # Evaluation hyperparameters from constants
    max_depth = SEVENSCENES_MAX_DEPTH
    sampling_number = SEVENSCENES_SAMPLING_NUMBER
    voxel_length = SEVENSCENES_VOXEL_LENGTH
    sdf_trunc = SEVENSCENES_SDF_TRUNC
    eval_threshold = SEVENSCENES_EVAL_THRESHOLD
    down_sample = SEVENSCENES_DOWN_SAMPLE

    # Fixed camera intrinsics for all 7Scenes images
    fx = SEVENSCENES_FX
    fy = SEVENSCENES_FY
    cx = SEVENSCENES_CX
    cy = SEVENSCENES_CY

    def __init__(self):
        super().__init__()
        self._scene_cache = {}

    # ------------------------------
    # Public API
    # ------------------------------

    def get_data(self, scene: str) -> Dict:
        """
        Collect per-view image paths, intrinsics/extrinsics for a scene.

        Args:
            scene: Scene identifier (e.g., "chess")

        Returns:
            Dict with:
                - image_files: List[str] - paths to images
                - extrinsics: np.ndarray [N, 4, 4] - world-to-camera transforms
                - intrinsics: np.ndarray [N, 3, 3] - camera intrinsics
                - aux: Dict with gt_mesh_path, gt_depth_files
        """
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        # Different sequence for stairs scene
        if scene == "stairs":
            data_folder = os.path.join(self.data_root, "7Scenes", scene, "seq-02")
            n_imgs = 500
        else:
            data_folder = os.path.join(self.data_root, "7Scenes", scene, "seq-01")
            n_imgs = 1000

        gt_mesh_path = os.path.join(self.data_root, "7Scenes", "meshes", f"{scene}.ply")

        # Fixed intrinsics for all images
        ixt = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float32)

        out = Dict({
            "image_files": [],
            "extrinsics": [],
            "intrinsics": [],
            "aux": Dict({
                "gt_mesh_path": gt_mesh_path,
                "gt_depth_files": [],
            }),
        })

        for i in range(0, n_imgs, 1):
            img_path = os.path.join(data_folder, f"frame-{i:06d}.color.png")
            pose_path = os.path.join(data_folder, f"frame-{i:06d}.pose.txt")
            depth_path = os.path.join(data_folder, f"frame-{i:06d}.depth.png")

            if not os.path.exists(img_path) or not os.path.exists(pose_path):
                continue

            # Load camera-to-world pose and convert to world-to-camera (extrinsic)
            c2w = np.loadtxt(pose_path)
            ext = np.linalg.inv(c2w).astype(np.float32)

            out.image_files.append(img_path)
            out.extrinsics.append(ext)
            out.intrinsics.append(ixt.copy())
            out.aux.gt_depth_files.append(depth_path)

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)

        print(f"[7Scenes] {scene}: {len(out.image_files)} images")

        self._scene_cache[scene] = out
        return out

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:
        """
        Evaluate fused point cloud against 7Scenes ground truth mesh.

        Args:
            scene: Scene identifier
            fuse_path: Path to fused point cloud (.ply)

        Returns:
            Dict with metrics: acc, comp, overall, precision, recall, fscore
        """
        gt_data = self.get_data(scene)
        gt_mesh_path = gt_data.aux.gt_mesh_path

        # Load and sample ground truth mesh
        gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        gt_pcd = sample_points_from_mesh(gt_mesh, self.sampling_number)

        # Load predicted point cloud
        pred_pcd = o3d.io.read_point_cloud(fuse_path)

        # Evaluate using shared utility function
        metrics = evaluate_3d_reconstruction(
            pred_pcd,
            gt_pcd,
            threshold=self.eval_threshold,
            down_sample=self.down_sample,
        )

        return metrics

    def _load_gt_meta(self, result_path: str) -> Dict:
        """
        Load saved GT meta (extrinsics, intrinsics, image_files) for fusion.

        This is needed when frames are sampled, so fuse3d uses the correct
        (sampled) GT instead of full dataset GT.

        Args:
            result_path: Path to npz file (used to derive gt_meta.npz path)

        Returns:
            Dict with GT data, or None if gt_meta.npz doesn't exist
        """
        export_dir = os.path.dirname(result_path)  # exports/mini_npz/
        gt_meta_path = os.path.join(os.path.dirname(export_dir), "gt_meta.npz")

        if os.path.exists(gt_meta_path):
            data = np.load(gt_meta_path, allow_pickle=True)
            # Build aux with gt_depth_files derived from image_files
            image_files = list(data["image_files"])
            gt_depth_files = [
                img_path.replace("color", "depth").replace(".color.", ".depth.")
                for img_path in image_files
            ]
            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
                "image_files": image_files,
                "aux": Dict({"gt_depth_files": gt_depth_files}),
            })
        return None

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:
        """
        Fuse per-view depths into a point cloud using TSDF fusion.

        Args:
            scene: Scene identifier
            result_path: Path to npz file with predicted depths/poses
            fuse_path: Output path for fused point cloud (.ply)
            mode: "recon_unposed" or "recon_posed"
        """
        # Try to load saved GT meta (handles frame sampling)
        gt_meta = self._load_gt_meta(result_path)
        if gt_meta is not None:
            gt_data = gt_meta
        else:
            gt_data = self.get_data(scene)
        _wait_for_file_ready(result_path)
        pred_data = Dict({k: v for k, v in np.load(result_path).items()})

        # Load original images (keep original size)
        images = []
        orig_sizes = []
        for img_path in gt_data.image_files:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            orig_sizes.append((img.shape[0], img.shape[1]))

        # Prepare depths, intrinsics, extrinsics
        if mode == "recon_unposed":
            depths, intrinsics, extrinsics = self._prep_unposed(
                pred_data, gt_data, orig_sizes, scene=scene
            )
        elif mode == "recon_posed":
            depths, intrinsics, extrinsics = self._prep_posed(
                pred_data, gt_data, orig_sizes, scene=scene
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        images = np.stack(images, axis=0)

        # Create TSDF volume and fuse
        volume = create_tsdf_volume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
        )
        mesh = fuse_depth_to_tsdf(
            volume, depths, images, intrinsics, extrinsics, max_depth=self.max_depth
        )

        # Sample points from mesh
        pcd = sample_points_from_mesh(mesh, self.sampling_number)

        # Save point cloud
        os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        o3d.io.write_point_cloud(fuse_path, pcd)

    # ------------------------------
    # Private helpers
    # ------------------------------

    def _prep_unposed(
        self, pred_data: Dict, gt_data: Dict, orig_sizes: list, scene: str
    ) -> tuple:
        """
        Prepare depths/intrinsics/extrinsics for recon_unposed mode.

        Similar to ETH3D but uses GT depth for masking instead of separate mask files.
        """
        # Scale alignment with fixed random_state for reproducibility
        _, _, scale, extrinsics = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )

        model_h, model_w = pred_data.depth.shape[1], pred_data.depth.shape[2]

        depths_out = []
        intrinsics_out = []
        for i in range(len(pred_data.depth)):
            orig_h, orig_w = orig_sizes[i]

            # Resize depth to original image size (nearest interpolation)
            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Load GT depth for masking
            gt_zero_mask = self._load_gt_mask(gt_data.aux.gt_depth_files[i])

            # Mask invalid depths BEFORE scale
            depth = self._mask_invalid_depth(depth, gt_zero_mask)

            # Apply scale AFTER mask
            depth = depth * scale

            # Adjust intrinsics to original image size
            h_ratio = orig_h / model_h
            w_ratio = orig_w / model_w
            ixt = pred_data.intrinsics[i].copy()
            ixt[0, :] *= w_ratio
            ixt[1, :] *= h_ratio

            depths_out.append(depth)
            intrinsics_out.append(ixt)

        return np.stack(depths_out), np.stack(intrinsics_out), extrinsics

    def _prep_posed(
        self, pred_data: Dict, gt_data: Dict, orig_sizes: list, scene: str
    ) -> tuple:
        """
        Prepare depths/intrinsics/extrinsics for recon_posed mode.
        Uses GT intrinsics/extrinsics but aligns depth scale via Umeyama.
        """
        # Scale alignment with fixed random_state
        _, _, scale, _ = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )

        model_h, model_w = pred_data.depth.shape[1], pred_data.depth.shape[2]

        depths_out = []
        for i in range(len(pred_data.depth)):
            orig_h, orig_w = orig_sizes[i]

            # Resize depth to original image size
            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Load GT depth for masking
            gt_zero_mask = self._load_gt_mask(gt_data.aux.gt_depth_files[i])

            # Mask invalid depths BEFORE scale
            depth = self._mask_invalid_depth(depth, gt_zero_mask)

            # Apply scale AFTER mask
            depth = depth * scale

            depths_out.append(depth)

        # Use GT intrinsics and extrinsics
        return np.stack(depths_out), gt_data.intrinsics.copy(), gt_data.extrinsics.copy()

    def _load_gt_mask(self, gt_depth_path: str) -> np.ndarray:
        """
        Load GT depth and create valid mask.

        For 7Scenes, GT depth is stored as 16-bit PNG in millimeters.
        Value 65535 indicates invalid depth.

        Returns:
            Boolean mask where True = valid region to keep
        """
        if not os.path.exists(gt_depth_path):
            return None

        gt_depth = cv2.imread(gt_depth_path, -1)
        if gt_depth is None:
            return None

        # 65535 is invalid depth marker in 7Scenes
        gt_depth[gt_depth == 65535] = 0
        # Convert to meters
        gt_depth = gt_depth / 1000.0

        # Valid mask: depth > 0
        valid_mask = gt_depth > 0
        return valid_mask

    def _mask_invalid_depth(
        self, depth: np.ndarray, gt_zero_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Mask invalid depth values by setting them to 0.

        Args:
            depth: Depth map to mask
            gt_zero_mask: Optional GT mask (True = valid region)

        Returns:
            Masked depth map with invalid regions set to 0
        """
        depth = depth.copy()

        if gt_zero_mask is not None:
            # Also mask out invalid pred depth
            pred_invalid = np.isnan(depth) | np.isinf(depth)
            combined_mask = np.logical_and(gt_zero_mask, np.logical_not(pred_invalid))
            depth = depth * combined_mask.astype(np.float32)
        else:
            # Fallback: only mask pred invalid values
            invalid_mask = np.isnan(depth) | np.isinf(depth) | (depth <= 0)
            depth[invalid_mask] = 0.0

        return depth


