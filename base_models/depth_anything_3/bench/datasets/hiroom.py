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
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/datasets/hiroom.py

"""
HiRoom Dataset implementation.

HiRoom is an indoor RGB-D dataset containing ground truth camera poses,
depth maps, and fused point clouds.

Evaluation metrics:
- 3D reconstruction: Accuracy, Completeness, F-score
- Camera pose estimation: AUC metrics
"""

import os
from typing import Dict as TDict, List

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
    HIROOM_DOWN_SAMPLE,
    HIROOM_EVAL_DATA_ROOT,
    HIROOM_EVAL_THRESHOLD,
    HIROOM_GT_ROOT_PATH,
    HIROOM_MAX_DEPTH,
    HIROOM_SAMPLING_NUMBER,
    HIROOM_SCENE_LIST_PATH,
    HIROOM_SDF_TRUNC,
    HIROOM_VOXEL_LENGTH,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama


def _load_scene_list() -> List[str]:
    """Load scene list from file."""
    if os.path.exists(HIROOM_SCENE_LIST_PATH):
        with open(HIROOM_SCENE_LIST_PATH, "r") as f:
            return f.read().splitlines()
    return []


@MV_REGISTRY.register(name="hiroom")
@MONO_REGISTRY.register(name="hiroom")
class HiRoomDataset(Dataset):
    """
    HiRoom Dataset wrapper for DepthAnything3 evaluation.

    Supports:
        - Camera pose estimation evaluation (AUC metrics)
        - 3D reconstruction evaluation (Accuracy, Completeness, F-score)
        - TSDF-based point cloud fusion

    Dataset structure:
        HiRoom/
        ├── {scene_path}/
        │   ├── image/           # RGB images
        │   ├── depth/           # GT depth maps
        │   ├── pose/            # Camera poses (.npy)
        │   ├── cam_K.npy        # Camera intrinsics
        │   └── aliasing_mask/   # Aliasing masks
        
        fused_pcd/
        └── {scene_name}.ply     # Ground truth fused point cloud
    """

    data_root = HIROOM_EVAL_DATA_ROOT
    gt_root_path = HIROOM_GT_ROOT_PATH
    SCENES = _load_scene_list()

    # Evaluation hyperparameters from constants
    max_depth = HIROOM_MAX_DEPTH
    sampling_number = HIROOM_SAMPLING_NUMBER
    voxel_length = HIROOM_VOXEL_LENGTH
    sdf_trunc = HIROOM_SDF_TRUNC
    eval_threshold = HIROOM_EVAL_THRESHOLD
    down_sample = HIROOM_DOWN_SAMPLE

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
            scene: Scene path (e.g., "xxx/yyy/zzz")

        Returns:
            Dict with:
                - image_files: List[str] - paths to images
                - extrinsics: np.ndarray [N, 4, 4] - world-to-camera transforms
                - intrinsics: np.ndarray [N, 3, 3] - camera intrinsics
                - aux: Dict with gt_pcd_path, gt_depth_files, aliasing_mask_files
        """
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = os.path.join(self.data_root, scene)
        image_dir = os.path.join(scene_dir, "image")

        # Get scene name for GT point cloud
        scene_name = "-".join(scene.split("/")[-3:])
        gt_pcd_path = os.path.join(self.gt_root_path, f"{scene_name}.ply")

        # Load shared camera intrinsics
        intrin_path = os.path.join(scene_dir, "cam_K.npy")
        ixt_shared = np.load(intrin_path).astype(np.float32)

        # Get all image names sorted
        image_names = sorted(os.listdir(image_dir))

        out = Dict({
            "image_files": [],
            "extrinsics": [],
            "intrinsics": [],
            "aux": Dict({
                "gt_pcd_path": gt_pcd_path,
                "gt_depth_files": [],
                "aliasing_mask_files": [],
            }),
        })

        for img_name in image_names:
            img_path = os.path.join(image_dir, img_name)
            frame_name = img_name.split(".")[0]

            # Depth and pose paths
            depth_path = os.path.join(scene_dir, "depth", f"{frame_name}.png")
            pose_path = os.path.join(scene_dir, "pose", f"{frame_name}.npy")
            aliasing_mask_path = os.path.join(scene_dir, "aliasing_mask", f"{frame_name}.png")

            if not os.path.exists(pose_path):
                continue

            # Load extrinsics (world-to-camera)
            ext = np.load(pose_path).astype(np.float32)

            out.image_files.append(img_path)
            out.extrinsics.append(ext)
            out.intrinsics.append(ixt_shared.copy())
            out.aux.gt_depth_files.append(depth_path)
            out.aux.aliasing_mask_files.append(aliasing_mask_path)

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)

        print(f"[HiRoom] {scene}: {len(out.image_files)} images")

        self._scene_cache[scene] = out
        return out

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:
        """
        Evaluate fused point cloud against HiRoom ground truth point cloud.

        Args:
            scene: Scene identifier
            fuse_path: Path to fused point cloud (.ply)

        Returns:
            Dict with metrics: acc, comp, overall, precision, recall, fscore
        """
        gt_data = self.get_data(scene)
        gt_pcd_path = gt_data.aux.gt_pcd_path

        # Load ground truth point cloud
        gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)

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
        """Load saved GT meta for fusion."""
        export_dir = os.path.dirname(result_path)
        gt_meta_path = os.path.join(os.path.dirname(export_dir), "gt_meta.npz")

        if os.path.exists(gt_meta_path):
            data = np.load(gt_meta_path, allow_pickle=True)
            image_files = list(data["image_files"])
            return Dict({
                "extrinsics": data["extrinsics"],
                "intrinsics": data["intrinsics"],
                "image_files": image_files,
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
        # Get full GT data
        full_gt_data = self.get_data(scene)

        # Try to load saved GT meta (handles frame sampling)
        gt_meta = self._load_gt_meta(result_path)
        if gt_meta is not None:
            gt_data = gt_meta
            image_indices = [
                full_gt_data.image_files.index(f)
                for f in gt_data.image_files
                if f in full_gt_data.image_files
            ]
        else:
            gt_data = full_gt_data
            image_indices = list(range(len(full_gt_data.image_files)))

        _wait_for_file_ready(result_path)
        pred_data = Dict({k: v for k, v in np.load(result_path).items()})

        # Load images
        images = []
        orig_sizes = []
        for img_idx in image_indices:
            img_path = full_gt_data.image_files[img_idx]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            orig_sizes.append((img.shape[0], img.shape[1]))

        images = np.stack(images, axis=0)

        # Prepare depths, intrinsics, extrinsics
        if mode == "recon_unposed":
            depths, intrinsics, extrinsics = self._prep_unposed(
                pred_data, gt_data, full_gt_data, image_indices, orig_sizes, scene=scene
            )
        elif mode == "recon_posed":
            depths, intrinsics, extrinsics = self._prep_posed(
                pred_data, gt_data, full_gt_data, image_indices, orig_sizes, scene=scene
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

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
        self, pred_data: Dict, gt_data: Dict, full_gt_data: Dict,
        image_indices: list, orig_sizes: list, scene: str = None
    ) -> tuple:
        """Prepare depths/intrinsics/extrinsics for recon_unposed mode."""
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
            img_idx = image_indices[i]

            # Resize depth to original image size
            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Load GT mask
            gt_zero_mask = self._load_gt_mask(
                full_gt_data.aux.gt_depth_files[img_idx],
                full_gt_data.aux.aliasing_mask_files[img_idx],
            )

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
        self, pred_data: Dict, gt_data: Dict, full_gt_data: Dict,
        image_indices: list, orig_sizes: list, scene: str = None
    ) -> tuple:
        """Prepare depths/intrinsics/extrinsics for recon_posed mode."""
        # Scale alignment
        _, _, scale, _ = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )

        depths_out = []
        for i in range(len(pred_data.depth)):
            orig_h, orig_w = orig_sizes[i]
            img_idx = image_indices[i]

            # Resize depth to original image size
            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Load GT mask
            gt_zero_mask = self._load_gt_mask(
                full_gt_data.aux.gt_depth_files[img_idx],
                full_gt_data.aux.aliasing_mask_files[img_idx],
            )

            # Mask invalid depths BEFORE scale
            depth = self._mask_invalid_depth(depth, gt_zero_mask)

            # Apply scale AFTER mask
            depth = depth * scale

            depths_out.append(depth)

        # Use GT intrinsics and extrinsics
        gt_intrinsics = np.stack([full_gt_data.intrinsics[idx] for idx in image_indices])
        gt_extrinsics = np.stack([full_gt_data.extrinsics[idx] for idx in image_indices])

        return np.stack(depths_out), gt_intrinsics, gt_extrinsics

    def _load_gt_mask(self, gt_depth_path: str, aliasing_mask_path: str) -> np.ndarray:
        """
        Load GT depth and aliasing mask to create valid mask.

        For HiRoom:
        - GT depth is stored as 16-bit PNG, scaled to 100m range
        - Aliasing mask marks regions to exclude

        Returns:
            Boolean mask where True = valid region to keep
        """
        # Load GT depth
        if os.path.exists(gt_depth_path):
            gt_depth = cv2.imread(gt_depth_path, -1) / 65535.0 * 100.0
        else:
            return None

        # Load aliasing mask
        aliasing_mask = None
        if os.path.exists(aliasing_mask_path):
            aliasing_mask = cv2.imread(aliasing_mask_path, -1) > 0

        # Valid mask: depth > 0 and not in aliasing region
        valid_mask = gt_depth > 0
        if aliasing_mask is not None:
            valid_mask = np.logical_and(valid_mask, np.logical_not(aliasing_mask))

        return valid_mask

    def _mask_invalid_depth(
        self, depth: np.ndarray, gt_zero_mask: np.ndarray = None
    ) -> np.ndarray:
        """Mask invalid depth values by setting them to 0."""
        depth = depth.copy()

        if gt_zero_mask is not None:
            pred_invalid = np.isnan(depth) | np.isinf(depth)
            combined_mask = np.logical_and(gt_zero_mask, np.logical_not(pred_invalid))
            depth = depth * combined_mask.astype(np.float32)
        else:
            invalid_mask = np.isnan(depth) | np.isinf(depth) | (depth <= 0)
            depth[invalid_mask] = 0.0

        return depth

