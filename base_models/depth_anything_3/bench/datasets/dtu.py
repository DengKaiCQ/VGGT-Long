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
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/datasets/dtu.py

"""
DTU Benchmark dataset implementation.

DTU is a multi-view stereo benchmark for 3D reconstruction evaluation.
Reference: https://roboimagedata.compute.dtu.dk/

Note: DepthAnything3 was never trained on any images from DTU.
"""

import glob
import os
from typing import Dict as TDict, List

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from addict import Dict
from PIL import Image
from plyfile import PlyData
from scipy.io import loadmat
from sklearn import neighbors as skln
from tqdm import tqdm

from depth_anything_3.bench.dataset import Dataset
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.utils.constants import (
    DTU_DIST_THRESH,
    DTU_EVAL_DATA_ROOT,
    DTU_MAX_POINTS,
    DTU_NUM_CONSIST,
    DTU_SCENES,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama


@MV_REGISTRY.register(name="dtu")
@MONO_REGISTRY.register(name="dtu")
class DTU(Dataset):
    """
    DTU Benchmark dataset wrapper for DepthAnything3 evaluation.

    Supports:
        - Camera pose estimation evaluation (AUC metrics)
        - 3D reconstruction evaluation (accuracy, completeness, overall)
        - Point cloud fusion from depth maps

    The dataset uses MVSNet evaluation protocol:
    https://drive.google.com/file/d/1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D/view
    """

    data_root = DTU_EVAL_DATA_ROOT
    SCENES = DTU_SCENES

    # Evaluation/triangulation hyperparameters from constants
    dist_thresh = DTU_DIST_THRESH
    num_consist = DTU_NUM_CONSIST

    # ------------------------------
    # Public API
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
        extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
        intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
        return intrinsics, extrinsics

    def get_data(self, scene: str) -> Dict:
        """
        Collect per-view image paths, intrinsics/extrinsics, and GT masks.

        Args:
            scene: Scene identifier (e.g., "scan1")

        Returns:
            Dict with:
                - image_files: List[str] - paths to images
                - extrinsics: np.ndarray [N, 4, 4]
                - intrinsics: np.ndarray [N, 3, 3]
                - aux.mask_files: List[str] - paths to depth masks
        """
        rgb_folder = os.path.join(self.data_root, "Rectified", scene)
        camera_folder = os.path.join(self.data_root, "Cameras")

        files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
        # Reorder: place index 33 first (reference view convention)
        files = [files[33]] + files[:33] + files[34:]

        out = Dict(
            {
                "image_files": files,
                "extrinsics": [],
                "intrinsics": [],
                "aux": Dict({"mask_files": []}),
            }
        )

        for rgb_file in files:
            basename = os.path.basename(rgb_file)
            file_idx = basename.split("_")[1]
            cam_idx = depth_idx = int(file_idx) - 1

            mask_file = self._depth_mask_path(scene, depth_idx)
            proj_mat_filename = os.path.join(camera_folder, f"{cam_idx:0>8}_cam.txt")

            ixt, ext = self.read_cam_file(proj_mat_filename)
            out.extrinsics.append(ext)
            out.intrinsics.append(ixt)
            out.aux.mask_files.append(mask_file)

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)
        return out

    def get_3dgtpath(self, scene: str) -> str:
        """Get path to ground truth point cloud for a scene."""
        scene_id = int(scene[4:])
        return os.path.join(self.data_root, f"Points/stl/stl{scene_id:03}_total.ply")

    def eval3d(self, scene: str, fuse_path: str, use_gpu: bool = False) -> TDict[str, float]:
        """
        Evaluate fused point cloud against DTU GT with ObsMask/Plane.

        Args:
            scene: Scene identifier
            fuse_path: Path to fused point cloud
            use_gpu: If True, use GPU-accelerated distance computation (faster but may have minor numerical differences)

        Returns:
            Dict with metrics: {"comp": float, "acc": float, "overall": float}
        """
        scene_id = int(scene[4:])
        gt_ply = os.path.join(self.data_root, f"Points/stl/stl{scene_id:03}_total.ply")
        mask_file = os.path.join(
            self.data_root, f"SampleSet/mvs_data/ObsMask/ObsMask{scene_id}_10.mat"
        )
        plane_file = os.path.join(
            self.data_root, f"SampleSet/mvs_data/ObsMask/Plane{scene_id}.mat"
        )
        result = self._evaluate_reconstruction(
            scene, fuse_path, gt_ply, mask_file, plane_file, use_gpu=use_gpu
        )
        return {"comp": result[0], "acc": result[1], "overall": result[2]}

    def load_masks(self, mask_files: List[str]) -> np.ndarray:
        """
        Load DTU depth validity masks.

        Args:
            mask_files: List of paths to mask images

        Returns:
            Boolean array [N, H, W] indicating valid depth regions
        """
        masks = []
        for mask_file in mask_files:
            mask = Image.open(mask_file)
            mask = np.array(mask, dtype=np.float32)
            masks.append(mask > 10)
        return np.asarray(masks)

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:
        """
        Fuse per-view depths into a point cloud and save to PLY.

        Args:
            scene: Scene identifier (e.g., "scan114")
            result_path: Path to npz file containing predicted depths/poses
            fuse_path: Output path for fused point cloud (.ply)
            mode: "recon_unposed" or "recon_posed"
        """
        gt_data = self.get_data(scene)
        pred_data = Dict({k: v for k, v in np.load(result_path).items()})
        masks = self.load_masks(gt_data.aux.mask_files)

        if mode == "recon_unposed":
            depths, intrinsics, extrinsics = self._prep_unposed(pred_data, gt_data, masks)
        elif mode == "recon_posed":
            depths, intrinsics, extrinsics = self._prep_posed(pred_data, gt_data, masks)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        proj_mat = self._build_proj_mats(intrinsics, extrinsics)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        depths_t = torch.from_numpy(depths).to(device=device, dtype=dtype).unsqueeze(1)
        proj_t = torch.from_numpy(proj_mat).to(device=device, dtype=dtype)
        height, width = depths_t.shape[-2:]

        points: List[np.ndarray] = []
        for idx in range(len(gt_data.image_files)):
            if mode == "recon_unposed":
                # Simple unfiltered back-projection per frame
                cur_p_pcd = self._generate_points_from_depth(
                    depths_t[idx : idx + 1], proj_t[idx : idx + 1]
                )
                mask = (depths_t[idx : idx + 1] > 0.001).squeeze()
                cur_p_pcd = cur_p_pcd[:, :, mask]
                no_filter_pc = cur_p_pcd.squeeze(0).permute(1, 0).cpu().numpy()
                points.append(no_filter_pc)
            else:  # recon_posed
                final_pc = self._fuse_consistent_points(depths_t, proj_t, idx, height, width)
                points.append(final_pc)

        # Concatenate and optionally downsample to hard cap
        points_np = np.concatenate(points, axis=0)
        points_np = self._cap_points(points_np, max_points=DTU_MAX_POINTS)

        os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        o3d.io.write_point_cloud(fuse_path, pcd)

    # ------------------------------
    # Geometry helpers
    # ------------------------------

    def _generate_points_from_depth(
        self, depth: torch.Tensor, proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Back-project depth map into 3D world coordinates.

        Args:
            depth: Depth tensor [B, 1, H, W]
            proj: Projection matrix [B, 4, 4] = [[K@R, K@t], [0,0,0,1]]

        Returns:
            Point cloud tensor [B, 3, H, W]
        """
        batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
        inv_proj = torch.inverse(proj)
        rot = inv_proj[:, :3, :3]
        trans = inv_proj[:, :3, 3:4]

        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=depth.device),
                torch.arange(0, width, dtype=torch.float32, device=depth.device),
            ],
            indexing="ij",
        )
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)
        rot_xyz = torch.matmul(rot, xyz)
        rot_depth_xyz = rot_xyz * depth.view(batch, 1, -1)
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)
        return proj_xyz.view(batch, 3, height, width)

    def _homo_warping(
        self,
        src_fea: torch.Tensor,
        src_proj: torch.Tensor,
        ref_proj: torch.Tensor,
        depth_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Homography warping for multi-view consistency checking.

        Args:
            src_fea: Source features [B, C, H, W]
            src_proj: Source projection [B, 4, 4]
            ref_proj: Reference projection [B, 4, 4]
            depth_values: Depth values [B, Ndepth] or [B, Ndepth, H, W]

        Returns:
            Warped features [B, C, H, W]
        """
        batch, channels = src_fea.shape[0], src_fea.shape[1]
        height, width = src_fea.shape[2], src_fea.shape[3]

        with torch.no_grad():
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]
            trans = proj[:, :3, 3:4]

            y, x = torch.meshgrid(
                [
                    torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                    torch.arange(0, width, dtype=torch.float32, device=src_fea.device),
                ],
                indexing="ij",
            )
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)
            xyz = torch.stack((x, y, torch.ones_like(x)))
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)
            rot_xyz = torch.matmul(rot, xyz)

            rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values.view(-1, 1, 1, height * width)
            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)

        warped_src_fea = F.grid_sample(
            src_fea,
            grid.view(batch, height, width, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return warped_src_fea.view(batch, channels, height, width)

    def _filter_depth(
        self,
        ref_depth: torch.Tensor,
        src_depths: torch.Tensor,
        ref_proj: torch.Tensor,
        src_projs: torch.Tensor,
    ) -> tuple:
        """
        Compute geometric consistency between reference and source depths.

        Args:
            ref_depth: Reference depth [1, 1, H, W]
            src_depths: Source depths [B, 1, H, W]
            ref_proj: Reference projection [1, 4, 4]
            src_projs: Source projections [B, 4, 4]

        Returns:
            Tuple of (ref_pc, aligned_pcs, dist)
        """
        ref_pc = self._generate_points_from_depth(ref_depth, ref_proj)
        src_pcs = self._generate_points_from_depth(src_depths, src_projs)
        aligned_pcs = self._homo_warping(src_pcs, src_projs, ref_proj, ref_depth)
        x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0]) ** 2
        y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1]) ** 2
        z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2]) ** 2
        dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)
        return ref_pc, aligned_pcs, dist

    def _extract_points(
        self, pc: torch.Tensor, mask: torch.Tensor, rgb: np.ndarray = None
    ) -> np.ndarray:
        """Extract masked points from a dense grid."""
        pc = pc.cpu().numpy()
        mask = mask.cpu().numpy().reshape(-1)
        pc = pc.reshape(-1, 3)
        points = pc[np.where(mask)]
        if rgb is not None:
            rgb = rgb.reshape(-1, 3)
            colors = rgb[np.where(mask)]
            return np.concatenate([points, colors], axis=1)
        return points

    # ------------------------------
    # 3D Reconstruction Evaluation
    # ------------------------------

    def _evaluate_reconstruction(
        self,
        scanid: str,
        pred_ply: str,
        gt_ply: str,
        mask_file: str,
        plane_file: str,
        down_dense: float = 0.2,
        patch: int = 60,
        max_dist: int = 20,
        use_gpu: bool = False,
    ) -> tuple:
        """
        Compute accuracy, completeness, and overall metrics for one scan.

        Args:
            scanid: Scan identifier
            pred_ply: Predicted point cloud path or array
            gt_ply: Ground truth point cloud path or array
            mask_file: ObsMask file path
            plane_file: Plane file path
            down_dense: Downsample density (min distance between points)
            patch: Patch size for boundary
            max_dist: Outlier threshold in mm
            use_gpu: If True, use GPU-accelerated distance computation

        Returns:
            Tuple of (mean_d2s, mean_s2d, overall)
        """
        thresh = down_dense

        # Load and downsample predicted point cloud
        data_pcd = self._read_ply(pred_ply) if isinstance(pred_ply, str) else pred_ply
        # Use fixed seed for reproducibility
        shuffle_rng = np.random.default_rng(seed=42)
        shuffle_rng.shuffle(data_pcd, axis=0)

        # Downsample point cloud
        nn_engine = skln.NearestNeighbors(
            n_neighbors=1, radius=thresh, algorithm="kd_tree", n_jobs=-1
        )
        nn_engine.fit(data_pcd)
        rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
        mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
        for curr, idxs in enumerate(rnn_idxs):
            if mask[curr]:
                mask[idxs] = 0
                mask[curr] = 1
        data_down = data_pcd[mask]

        # Restrict to observed volume (ObsMask)
        obs_mask_file = loadmat(mask_file)
        ObsMask, BB, Res = (obs_mask_file[attr] for attr in ["ObsMask", "BB", "Res"])
        BB = BB.astype(np.float32)

        inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(
            axis=-1
        ) == 3
        data_in = data_down[inbound]

        data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
        grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(
            axis=-1
        ) == 3
        data_grid_in = data_grid[grid_inbound]
        in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(
            np.bool_
        )
        data_in_obs = data_in[grid_inbound][in_obs]

        # Compute accuracy (pred -> GT) and completeness (GT -> pred)
        stl = self._read_ply(gt_ply) if isinstance(gt_ply, str) else gt_ply

        if use_gpu and torch.cuda.is_available():
            # GPU-accelerated distance computation
            mean_d2s = self._knn_dist_gpu(data_in_obs, stl, max_dist)
        else:
            # CPU version (original, for exact reproduction)
            nn_engine.fit(stl)
            dist_d2s, _ = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
            mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

        ground_plane = loadmat(plane_file)["P"]
        stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
        above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
        stl_above = stl[above]

        if use_gpu and torch.cuda.is_available():
            # GPU-accelerated distance computation
            mean_s2d = self._knn_dist_gpu(stl_above, data_in, max_dist)
        else:
            # CPU version (original, for exact reproduction)
            nn_engine.fit(data_in)
            dist_s2d, _ = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
            mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

        overall = (mean_d2s + mean_s2d) / 2
        return mean_d2s, mean_s2d, overall

    def _knn_dist_gpu(
        self,
        query: np.ndarray,
        target: np.ndarray,
        max_dist: float,
        batch_size: int = 8192,
        target_batch_size: int = 50000,
    ) -> float:
        """
        GPU-accelerated nearest neighbor distance computation.

        Args:
            query: Query points [N, 3]
            target: Target points [M, 3]
            max_dist: Outlier threshold
            batch_size: Batch size for query to avoid OOM (tuned for 16GB GPU)
            target_batch_size: Batch size for target to avoid OOM

        Returns:
            Mean distance (excluding outliers)
        """
        device = torch.device("cuda")

        all_min_dists = []
        n_query_batches = (len(query) + batch_size - 1) // batch_size
        n_target_batches = (len(target) + target_batch_size - 1) // target_batch_size

        # Pre-load target batches to GPU to avoid repeated transfers
        # Memory: ~50000 pts * 3 coords * 4 bytes * n_batches
        target_batches = []
        for j in range(0, len(target), target_batch_size):
            target_batch = target[j : j + target_batch_size]
            target_t = torch.from_numpy(target_batch).float().to(device)
            target_batches.append(target_t)

        with tqdm(total=n_query_batches, desc="  GPU KNN", leave=False, ncols=100) as pbar:
            for i in range(0, len(query), batch_size):
                batch = query[i : i + batch_size]
                query_t = torch.from_numpy(batch).float().to(device)

                # Compute distances to all target batches
                # Memory peak: query_batch × target_batch_size × 4 bytes
                # = 8192 × 50000 × 4 = ~1.6 GB per cdist call
                batch_min_dists = []
                for target_t in target_batches:
                    dists = torch.cdist(query_t, target_t)
                    batch_min_dists.append(dists.min(dim=1).values)
                    del dists  # Free immediately

                # Get minimum distance across all target batches
                min_dists = torch.stack(batch_min_dists, dim=1).min(dim=1).values
                all_min_dists.append(min_dists.cpu().numpy())

                del query_t, min_dists, batch_min_dists
                pbar.update(1)

        # Clean up target batches
        for target_t in target_batches:
            del target_t
        torch.cuda.empty_cache()

        all_min_dists = np.concatenate(all_min_dists)
        return all_min_dists[all_min_dists < max_dist].mean()

    def _read_ply(self, file: str) -> np.ndarray:
        """Read point cloud from PLY file."""
        data = PlyData.read(file)
        vertex = data["vertex"]
        return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

    # ------------------------------
    # Private helpers
    # ------------------------------

    def _depth_mask_path(self, scene: str, depth_idx: int) -> str:
        """Get path to depth mask for a scene and frame."""
        return os.path.join(
            self.data_root, "depth_raw", "Depths", scene, f"depth_visual_{depth_idx:04d}.png"
        )

    def _prep_unposed(
        self, pred_data: Dict, gt_data: Dict, masks: np.ndarray
    ) -> tuple:
        """
        Prepare depths/intrinsics/extrinsics for recon_unposed mode.
        
        Applies Umeyama scale, rescales intrinsics if depth resolution differs,
        and zeroes invalid-mask depths (nearest interpolation as in paper).
        """
        _, _, scale, extrinsics = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            ransac=True,
            return_aligned=True,
            random_state=42,
        )
        depths = pred_data.depth * scale
        intrinsics = pred_data.intrinsics.copy()

        if depths.shape[-2:] != masks.shape[-2:]:
            # When resizing depths to mask size, adjust intrinsics accordingly
            sx = masks.shape[-1] / depths.shape[-1]
            sy = masks.shape[-2] / depths.shape[-2]
            intrinsics[:, 0:1] *= sx
            intrinsics[:, 1:2] *= sy
            depths = F.interpolate(
                torch.from_numpy(depths)[None].float(),
                size=(masks.shape[-2], masks.shape[-1]),
                mode="nearest",
            )[0].numpy()
            depths[masks == False] = 0.0  # noqa: E712

        return depths, intrinsics, extrinsics

    def _prep_posed(
        self, pred_data: Dict, gt_data: Dict, masks: np.ndarray
    ) -> tuple:
        """
        Prepare depths/intrinsics/extrinsics for recon_posed mode.

        Uses GT intrinsics/extrinsics but aligns scale via Umeyama.
        Same mask order as other datasets: mask BEFORE scale.
        """
        _, _, scale, _ = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            ransac=True,
            return_aligned=True,
            random_state=42,
        )
        depths = pred_data.depth.copy()
        intrinsics = gt_data.intrinsics.copy()
        extrinsics = gt_data.extrinsics.copy()

        if depths.shape[-2:] != masks.shape[-2:]:
            depths = F.interpolate(
                torch.from_numpy(depths)[None].float(),
                size=(masks.shape[-2], masks.shape[-1]),
                mode="nearest",
            )[0].numpy()

        # Mask BEFORE scale (same as other datasets)
        depths[masks == False] = 0.0  # noqa: E712
        depths = depths * scale

        return depths, intrinsics, extrinsics

    def _build_proj_mats(
        self, intrinsics: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """Compute per-view 4x4 projection matrices from K and [R|t]."""
        proj_mat_list = []
        for i in range(len(intrinsics)):
            proj_mat = np.eye(4, dtype=np.float32)
            proj_mat[:3, :4] = np.dot(intrinsics[i], extrinsics[i][:3])
            proj_mat_list.append(proj_mat)
        return np.stack(proj_mat_list, axis=0)

    def _fuse_consistent_points(
        self,
        depths_t: torch.Tensor,
        proj_t: torch.Tensor,
        idx: int,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Fuse points consistent across multiple source views for a reference index."""
        device, dtype = depths_t.device, depths_t.dtype
        pc_buff = torch.zeros((3, H, W), device=device, dtype=dtype)
        val_cnt = torch.zeros((1, H, W), device=device, dtype=dtype)

        j = 0
        batch_size = 20
        tot_frame = depths_t.shape[0]
        while True:
            ref_pc, pcs, dist = self._filter_depth(
                ref_depth=depths_t[idx : idx + 1],
                src_depths=depths_t[j : min(j + batch_size, tot_frame)],
                ref_proj=proj_t[idx : idx + 1],
                src_projs=proj_t[j : min(j + batch_size, tot_frame)],
            )
            masks = (dist < self.dist_thresh).float()
            masked_pc = pcs * masks
            pc_buff += masked_pc.sum(dim=0, keepdim=False)
            val_cnt += masks.sum(dim=0, keepdim=False)
            j += batch_size
            if j >= tot_frame:
                break

        final_mask = (val_cnt >= self.num_consist).squeeze(0)
        avg_points = torch.div(pc_buff, val_cnt).permute(1, 2, 0)
        final_pc = self._extract_points(avg_points, final_mask)
        return final_pc

    def _cap_points(self, points: np.ndarray, max_points: int) -> np.ndarray:
        """Downsample points if exceeding max count."""
        if len(points) <= max_points:
            return points
        # Use fixed seed for reproducibility
        rng = np.random.default_rng(seed=42)
        random_idx = rng.choice(len(points), max_points, replace=False)
        return points[random_idx]

