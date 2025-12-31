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
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/datasets/eth3d.py

"""
ETH3D Benchmark dataset implementation.

ETH3D is a multi-view stereo benchmark with high-resolution images and
accurate ground truth geometry from laser scanning.
Reference: https://www.eth3d.net/

Evaluation metrics:
- 3D reconstruction: Accuracy, Completeness, F-score
- Camera pose estimation: AUC metrics
"""

import glob
import os
from typing import Dict as TDict, List, Optional

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from addict import Dict
from PIL import Image

from depth_anything_3.bench.dataset import Dataset, _wait_for_file_ready
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    evaluate_3d_reconstruction,
    fuse_depth_to_tsdf,
    quat2rotmat,
    sample_points_from_mesh,
)
from depth_anything_3.utils.constants import (
    ETH3D_DOWN_SAMPLE,
    ETH3D_EVAL_DATA_ROOT,
    ETH3D_EVAL_THRESHOLD,
    ETH3D_FILTER_KEYS,
    ETH3D_MAX_DEPTH,
    ETH3D_SAMPLING_NUMBER,
    ETH3D_SCENES,
    ETH3D_SDF_TRUNC,
    ETH3D_VOXEL_LENGTH,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama


@MV_REGISTRY.register(name="eth3d")
@MONO_REGISTRY.register(name="eth3d")
class ETH3D(Dataset):
    """
    ETH3D Benchmark dataset wrapper for DepthAnything3 evaluation.

    Supports:
        - Camera pose estimation evaluation (AUC metrics)
        - 3D reconstruction evaluation (Accuracy, Completeness, F-score)
        - TSDF-based point cloud fusion

    Dataset structure:
        eth3d/multiview/
        ├── scene_name/
        │   ├── images/                    # RGB images
        │   ├── dslr_calibration_jpg/
        │   │   ├── cameras.txt            # Camera intrinsics
        │   │   └── images.txt             # Camera poses
        │   ├── combined_mesh.ply          # Ground truth mesh
        │   └── ground_truth_depth/        # GT depth maps (optional)
    """

    data_root = ETH3D_EVAL_DATA_ROOT
    SCENES = ETH3D_SCENES

    # Evaluation hyperparameters from constants
    max_depth = ETH3D_MAX_DEPTH
    sampling_number = ETH3D_SAMPLING_NUMBER
    voxel_length = ETH3D_VOXEL_LENGTH
    sdf_trunc = ETH3D_SDF_TRUNC
    eval_threshold = ETH3D_EVAL_THRESHOLD
    down_sample = ETH3D_DOWN_SAMPLE

    def __init__(self):
        super().__init__()
        # Pre-load scene data for efficiency
        self._scene_cache = {}

    # ------------------------------
    # Camera file parsing
    # ------------------------------

    def _parse_cameras_txt(self, filepath: str) -> dict:
        """
        Parse COLMAP-style cameras.txt file.

        Returns:
            Dict mapping camera_id to intrinsic parameters
        """
        camera_dict = {}
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines[3:]:  # Skip header
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                cam_id = parts[0]
                # Format: ID, MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, [distortion params...]
                camera_dict[cam_id] = {
                    "width": float(parts[2]),
                    "height": float(parts[3]),
                    "fx": float(parts[4]),
                    "fy": float(parts[5]),
                    "cx": float(parts[6]),
                    "cy": float(parts[7]),
                }
        return camera_dict

    def _parse_images_txt(self, filepath: str) -> dict:
        """
        Parse COLMAP-style images.txt file.

        Returns:
            Dict mapping image path to pose parameters
        """
        pose_dict = {}
        with open(filepath) as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[4:]):  # Skip header
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Every other line contains pose info
                if idx % 2 == 0:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                    image_id = parts[0]
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    camera_id = parts[8]
                    name = parts[9]
                    pose_dict[name] = {
                        "image_id": image_id,
                        "quat": [qw, qx, qy, qz],
                        "trans": [tx, ty, tz],
                        "camera_id": camera_id,
                    }
        return pose_dict

    def _should_filter_image(self, scene: str, image_name: str) -> bool:
        """Check if image should be filtered out based on known problematic views."""
        filter_keys = ETH3D_FILTER_KEYS.get(scene, [])
        for key in filter_keys:
            if image_name.endswith(key):
                return True
        return False

    # ------------------------------
    # Public API
    # ------------------------------

    def get_data(self, scene: str) -> Dict:
        """
        Collect per-view image paths, intrinsics/extrinsics for a scene.

        Args:
            scene: Scene identifier (e.g., "courtyard")

        Returns:
            Dict with:
                - image_files: List[str] - paths to images
                - extrinsics: np.ndarray [N, 4, 4] - world-to-camera transforms
                - intrinsics: np.ndarray [N, 3, 3] - camera intrinsics
                - aux: Dict with gt_mesh_path
        """
        # Check cache
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = os.path.join(self.data_root, scene)

        # Parse camera files
        cameras_file = os.path.join(scene_dir, "dslr_calibration_jpg", "cameras.txt")
        images_file = os.path.join(scene_dir, "dslr_calibration_jpg", "images.txt")
        camera_dict = self._parse_cameras_txt(cameras_file)
        pose_dict = self._parse_images_txt(images_file)

        # Ground truth mesh path
        gt_mesh_path = os.path.join(scene_dir, "combined_mesh.ply")

        out = Dict({
            "image_files": [],
            "extrinsics": [],
            "intrinsics": [],
            "aux": Dict({
                "gt_mesh_path": gt_mesh_path,
                "heights": [],
                "widths": [],
            }),
        })

        # Process each image (preserve original order from images.txt)
        filtered_count = 0
        for image_name, pose_info in pose_dict.items():
            # Filter problematic views
            if self._should_filter_image(scene, image_name):
                filtered_count += 1
                continue

            image_path = os.path.join(scene_dir, "images", image_name)
            if not os.path.exists(image_path):
                continue

            cam_info = camera_dict.get(pose_info["camera_id"])
            if cam_info is None:
                continue

            # Build intrinsics matrix
            ixt = np.array([
                [cam_info["fx"], 0, cam_info["cx"]],
                [0, cam_info["fy"], cam_info["cy"]],
                [0, 0, 1],
            ], dtype=np.float32)

            # Build extrinsics matrix (world-to-camera)
            # COLMAP format: world point -> camera point
            rot = quat2rotmat(pose_info["quat"])
            ext = np.eye(4, dtype=np.float32)
            ext[:3, :3] = rot
            ext[:3, 3] = pose_info["trans"]

            out.image_files.append(image_path)
            out.extrinsics.append(ext)
            out.intrinsics.append(ixt)
            out.aux.heights.append(cam_info["height"])
            out.aux.widths.append(cam_info["width"])

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)

        # Print scene info
        total_images = len(pose_dict)
        used_images = len(out.image_files)
        print(f"[ETH3D] {scene}: {used_images}/{total_images} images "
              f"(filtered {filtered_count}, missing {total_images - used_images - filtered_count})")
        
        if used_images < 3:
            print(f"[ETH3D] ⚠️  WARNING: {scene} has only {used_images} images - evaluation may fail!")

        # Cache result
        self._scene_cache[scene] = out
        return out

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:
        """
        Evaluate fused point cloud against ETH3D ground truth mesh.

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
        # gt_meta.npz is in the same exports/ directory as results.npz
        export_dir = os.path.dirname(result_path)  # exports/mini_npz/
        gt_meta_path = os.path.join(os.path.dirname(export_dir), "gt_meta.npz")

        if os.path.exists(gt_meta_path):
            data = np.load(gt_meta_path, allow_pickle=True)
            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
                "image_files": data["image_files"] if "image_files" in data else None,
            })
        return None

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:
        """
        Fuse per-view depths into a point cloud using TSDF fusion.

        Pipeline:
        1. Load original images (keep original size)
        2. Resize depth to original image size (nearest interpolation)
        3. Adjust intrinsics to original image size
        4. Apply scale alignment and mask invalid depths
        5. TSDF fusion

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
        orig_sizes = []  # (H, W) for each image
        for img_path in gt_data.image_files:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            orig_sizes.append((img.shape[0], img.shape[1]))

        # Prepare depths, intrinsics, extrinsics with resize to original size
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
        self, pred_data: Dict, gt_data: Dict, orig_sizes: list, scene: str = None
    ) -> tuple:
        """
        Prepare depths/intrinsics/extrinsics for recon_unposed mode.

        Pipeline:
        1. Umeyama scale alignment
        2. Load GT mask for each frame
        3. Resize depth to original image size (nearest)
        4. Apply GT mask BEFORE scale
        5. Apply scale
        6. Adjust intrinsics to original image size
        """
        # Scale alignment with fixed random_state for reproducibility
        _, _, scale, extrinsics = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )

        # Get model output size
        model_h, model_w = pred_data.depth.shape[1], pred_data.depth.shape[2]

        # Process each frame
        depths_out = []
        intrinsics_out = []
        for i in range(len(pred_data.depth)):
            orig_h, orig_w = orig_sizes[i]
            image_name = os.path.basename(gt_data.image_files[i])

            # Resize depth to original image size (nearest interpolation)
            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Load GT mask (apply BEFORE scale)
            gt_zero_mask = None
            if scene is not None:
                gt_zero_mask = self._load_gt_mask(scene, image_name, (orig_h, orig_w))

            # Mask invalid depths BEFORE scale
            depth = self._mask_invalid_depth(depth, gt_zero_mask)

            # Apply scale AFTER mask
            depth = depth * scale

            # Adjust intrinsics to original image size
            h_ratio = orig_h / model_h
            w_ratio = orig_w / model_w
            ixt = pred_data.intrinsics[i].copy()
            ixt[0, :] *= w_ratio  # fx, 0, cx
            ixt[1, :] *= h_ratio  # 0, fy, cy

            depths_out.append(depth)
            intrinsics_out.append(ixt)

        return np.stack(depths_out), np.stack(intrinsics_out), extrinsics

    def _prep_posed(
        self, pred_data: Dict, gt_data: Dict, orig_sizes: list, scene: str = None
    ) -> tuple:
        """
        Prepare depths/intrinsics/extrinsics for recon_posed mode.

        Uses GT intrinsics/extrinsics but aligns depth scale via Umeyama.
        Depth is resized to original image size.
        """
        # Scale alignment with fixed random_state for reproducibility
        _, _, scale, _ = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )

        # Process each frame
        depths_out = []
        for i in range(len(pred_data.depth)):
            orig_h, orig_w = orig_sizes[i]
            image_name = os.path.basename(gt_data.image_files[i])

            # Resize depth to original image size (nearest interpolation)
            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Load GT mask (apply BEFORE scale)
            gt_zero_mask = None
            if scene is not None:
                gt_zero_mask = self._load_gt_mask(scene, image_name, (orig_h, orig_w))

            # Mask invalid depths BEFORE scale
            depth = self._mask_invalid_depth(depth, gt_zero_mask)

            # Apply scale AFTER mask
            depth = depth * scale

            depths_out.append(depth)

        # Use GT intrinsics and extrinsics (already at original image size)
        return np.stack(depths_out), gt_data.intrinsics.copy(), gt_data.extrinsics.copy()

    def _load_gt_mask(self, scene: str, image_name: str, shape: tuple) -> np.ndarray:
        """
        Load GT mask for masking invalid regions.

        GT mask marks occluded or invalid regions that should be excluded
        from depth fusion and evaluation.

        Args:
            scene: Scene identifier
            image_name: Image filename (e.g., "DSC_0307.JPG")
            shape: (height, width) of the image

        Returns:
            Boolean mask where True = valid region to keep
        """
        h, w = shape

        # GT mask file path
        gt_mask_path = os.path.join(
            self.data_root, scene, "masks_for_images", "dslr_images",
            image_name.replace(".JPG", ".png")
        )

        # GT depth file path (used to determine valid depth regions)
        gt_depth_path = os.path.join(
            self.data_root, scene, "ground_truth_depth", "dslr_images", image_name
        )

        # Load GT depth
        if os.path.exists(gt_depth_path):
            gt_depth = np.fromfile(gt_depth_path, dtype=np.float32).reshape(h, w)
        else:
            gt_depth = np.ones((h, w), dtype=np.float32)

        # Load GT mask
        if os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = np.asarray(gt_mask)
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        # Compute zero_mask
        # gt_mask == 1 means occluded/invalid region
        invalid_mask_from_gt = gt_mask == 1
        gt_depth_copy = gt_depth.copy()
        gt_depth_copy[gt_mask == 1] = 0

        invalid_mask_from_gt_depth = np.logical_or(gt_depth_copy == 0, gt_depth_copy == np.inf)

        # zero_mask: valid region that should be kept
        zero_mask = np.logical_and(
            np.logical_not(invalid_mask_from_gt),
            np.logical_not(invalid_mask_from_gt_depth)
        )

        return zero_mask

    def _mask_invalid_depth(
        self, depth: np.ndarray, gt_zero_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Mask invalid depth values by setting them to 0.

        Logic:
        1. Apply GT mask (if provided) - marks occluded/invalid regions
        2. Mask pred invalid values (nan, inf)

        Args:
            depth: Depth map to mask
            gt_zero_mask: Optional GT mask (True = valid region)

        Returns:
            Masked depth map with invalid regions set to 0
        """
        depth = depth.copy()

        # Apply GT mask first (before scale)
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

