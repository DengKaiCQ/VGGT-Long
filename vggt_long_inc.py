import glob
import os
import shutil
import gc
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
base_models_path = os.path.join(current_dir, 'base_models')
if base_models_path not in sys.path:
    sys.path.append(base_models_path)

try:
    import onnxruntime
except ImportError:
    pass

from loop_utils.config_utils import load_config
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    apply_sim3_direct,
    compute_sim3_ab,
    save_confident_pointcloud_batch,
    weighted_align_point_maps,
)
from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import merge_ply_files
from loop_utils.sim3utils import warmup_numba
from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from base_models.base_model import VGGTAdapter, Pi3Adapter, MapAnythingAdapter, DA3Adapter

from LoopModels.LoopModel import LoopDetector
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW


# NOTE: This file is a resume-friendly variant of vggt_long.py.
# It can continue from existing cached chunks and emits callbacks after each
# chunk alignment and final global alignment.


def remove_duplicates(data_list):
    seen = {}
    result = []
    for item in data_list:
        if item[0] == item[2]:
            continue
        key = (item[0], item[2])
        if key not in seen:
            seen[key] = True
            result.append(item)
    return result


class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []
        self.all_camera_intrinsics = []


class VGGT_Long_Inc:
    def __init__(
        self,
        image_dir: str,
        save_dir: str,
        config: dict,
        on_chunk_aligned: Optional[Callable[[int], None]] = None,
        on_all_aligned: Optional[Callable[[], None]] = None,
        retrieval_instance: Optional[object] = None, # Allow passing persistent retrieval object
    ):
        self.config = config
        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )
        self.sky_mask = False
        self.useDBoW = self.config["Model"]["useDBoW"]

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_intrinsics = []

        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        self.chunk_indices = None
        self.loop_list = []
        self.loop_optimizer = Sim3LoopOptimizer(self.config)
        self.sim3_list = []
        self.loop_sim3_list = []
        self.loop_predict_list = []
        self.loop_enable = self.config["Model"]["loop_enable"]
        self.on_chunk_aligned = on_chunk_aligned
        self.on_all_aligned = on_all_aligned
        
        self.retrieval = retrieval_instance # Use passed instance if available

        # Initialize loop detector if not provided but needed
        # Note: self.retrieval holds either RetrievalDBOW or LoopDetector instance
        if self.loop_enable and self.retrieval is None:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.retrieval = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config
                )
                # Important: Initialize empty state for DNIO manually if fresh
                self.retrieval.image_paths = []
                self.retrieval.descriptors = None
        elif self.loop_enable:
            print("Reusing existing Loop Detector/Retrieval instance.")

        print("Loading model...")
        if self.config['Weights']['model'] == 'VGGT':
            self.model = VGGTAdapter(self.config)
        elif self.config['Weights']['model'] == 'Pi3':
            self.model = Pi3Adapter(self.config)
        elif self.config['Weights']['model'] == 'Mapanything':
            self.model = MapAnythingAdapter(self.config)
        elif self.config['Weights']['model'] == 'DA3':
            self.model = DA3Adapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}")
        
        self.model.load()
        print("init done.")

    # -------------------- helpers --------------------
    def _state_path(self):
        return os.path.join(self.output_dir, "state_meta.npz")

    def save_state(self, sim3_rel: List, sim3_abs: List, chunk_indices: List[Tuple[int, int]]):
        np.savez(
            self._state_path(),
            sim3_rel=np.array(sim3_rel, dtype=object),
            sim3_abs=np.array(sim3_abs, dtype=object),
            chunk_indices=np.array(chunk_indices, dtype=object),
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            num_images=len(self.img_list),
        )

    def load_state(self):
        path = self._state_path()
        if not os.path.exists(path):
            return None
        meta = np.load(path, allow_pickle=True)
        return {
            "sim3_rel": list(meta.get("sim3_rel", [])),
            "sim3_abs": list(meta.get("sim3_abs", [])),
            "chunk_indices": list(meta.get("chunk_indices", [])),
            "chunk_size": int(meta.get("chunk_size", self.chunk_size)),
            "overlap": int(meta.get("overlap", self.overlap)),
            "num_images": int(meta.get("num_images", 0)),
        }

    def count_existing_chunks(self) -> int:
        files = glob.glob(os.path.join(self.result_unaligned_dir, "chunk_*.npy"))
        if not files:
            files = glob.glob(os.path.join(self.result_aligned_dir, "chunk_*.npy"))
        return len(files)
    
    def get_loop_pairs(self):
        self.loop_list = []
        
        # Prepare full image paths
        # self.img_list is already updated in run()
        
        if self.useDBoW: # DBoW2
            skipped_count = 0
            processed_count = 0
            
            for frame_id, img_path in tqdm(enumerate(self.img_list), desc="Loop Detection"):
                # Optimization: if frame already in DB, skip loading image
                if hasattr(self.retrieval, 'stored_indices') and \
                   frame_id < len(self.retrieval.stored_indices) and \
                   self.retrieval.stored_indices[frame_id]:
                    skipped_count += 1
                    continue

                processed_count += 1
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)
                frame = cv2.resize(image_ori, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(
                    thresh=self.config['Loop']['DBoW']['thresh'], 
                    num_repeat=self.config['Loop']['DBoW']['num_repeat']
                )

                if cands is not None:
                    (i, j) = cands 
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)
                
                self.retrieval.save_up_to(frame_id)
            
            if hasattr(self.retrieval, 'prev_loop_closes'):
                for lc in self.retrieval.prev_loop_closes:
                    if lc not in self.loop_list:
                        self.loop_list.append(lc)
            
            print(f"DBoW: Skipped {skipped_count}, processed {processed_count} frames.")
            
        else: # DNIO (LoopDetector) - Incremental HACK
            detector = self.retrieval
            
            # Use Path objects for comparison as LoopDetector uses them internally
            from pathlib import Path
            current_paths = [Path(p) for p in self.img_list]
            
            # Identify new images
            old_paths_set = set(detector.image_paths) if detector.image_paths else set()
            new_paths = [p for p in current_paths if p not in old_paths_set]
            
            print(f"DNIO: Found {len(detector.image_paths) if detector.image_paths else 0} cached, {len(new_paths)} new images.")
            
            if len(new_paths) > 0:
                # Store old state
                old_descriptors = detector.descriptors
                
                # Temporarily set detector to process ONLY new paths
                detector.image_paths = new_paths
                
                # Run extraction on new images
                new_descriptors = detector.extract_descriptors()
                
                # Merge descriptors
                if old_descriptors is not None:
                    detector.descriptors = torch.cat([old_descriptors, new_descriptors])
                else:
                    detector.descriptors = new_descriptors
            
            # Restore full path list for search
            detector.image_paths = current_paths
            
            # Run search on full database (fast part)
            self.loop_list = detector.find_loop_closures()
            detector.save_results()
            
        print(f"Total loop pairs for optimization: {len(self.loop_list)}")

    # -------------------- core --------------------
    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            s2, e2 = range_2
            chunk_image_paths += self.img_list[s2:e2]

        predictions = self.model.infer_chunk(chunk_image_paths)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        # Save logic
        if is_loop:
             save_dir = self.result_loop_dir
             filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"
            
            extrinsics = predictions["extrinsic"]
            intrinsics = predictions["intrinsic"]
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        save_path = os.path.join(save_dir, filename)
        if 'depth' in predictions and predictions['depth'] is not None:
            predictions["depth"] = np.squeeze(predictions["depth"])
        np.save(save_path, predictions)
        
        return predictions if is_loop or range_2 is not None else None

    def process_long_sequence(self, resume: bool = False):
        step = self.chunk_size - self.overlap

        # resume metadata
        meta = self.load_state() if resume else None
        if meta and (meta.get("chunk_size") != self.chunk_size or meta.get("overlap") != self.overlap):
            raise ValueError("Chunk size/overlap changed; cannot resume safely.")
        prev_chunk_indices = list(meta["chunk_indices"]) if meta else []
        if resume and meta is None:
            print("Resume requested but no previous state found; running from scratch.")
        existing_chunks = len(prev_chunk_indices) if (resume and meta) else 0
        sim3_rel = meta["sim3_rel"] if meta else []

        # rebuild chunk indices
        if resume and prev_chunk_indices:
            self.chunk_indices = list(prev_chunk_indices)
            last_end = prev_chunk_indices[-1][1]
            total_len = len(self.img_list)
            while last_end < total_len:
                start_idx = max(0, last_end - self.overlap)
                end_idx = min(start_idx + self.chunk_size, total_len)
                self.chunk_indices.append((start_idx, end_idx))
                last_end = end_idx
        else:
            if len(self.img_list) <= self.chunk_size:
                self.chunk_indices = [(0, len(self.img_list))]
            else:
                num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
                self.chunk_indices = []
                for i in range(num_chunks):
                    s_idx = i * step
                    e_idx = min(s_idx + self.chunk_size, len(self.img_list))
                    self.chunk_indices.append((s_idx, e_idx))

        num_chunks = len(self.chunk_indices)
        start_chunk = existing_chunks if resume else 0
        print(
            f"Processing {len(self.img_list)} images in {num_chunks} chunks "
            f"with overlap {self.overlap}; resume start at chunk {start_chunk}"
        )

        # Trigger an initial update if in resume mode
        if resume and existing_chunks > 0 and self.on_all_aligned:
            print("Triggering initial refresh for existing cache...")
            self.on_all_aligned()

        # -------------------------------------------------------
        # Process and Align SEQUENTIALLY
        # -------------------------------------------------------
        for chunk_idx in range(start_chunk, len(self.chunk_indices)):
            rng = self.chunk_indices[chunk_idx]
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)} image ID: {rng[0]}-{rng[1]}")
            
            # 1. Process single chunk (generate unaligned)
            self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()
            
            # 2. Align with previous chunk immediately (if applicable)
            # Check if we have a previous chunk to align with
            # sim3_rel has length N-1 for N chunks.
            # chunk_idx starts at 0.
            # If chunk_idx > 0, we need sim3_rel[chunk_idx - 1] to exist.
            if chunk_idx > 0:
                rel_idx = chunk_idx - 1
                if rel_idx >= len(sim3_rel):
                    # Perform alignment
                    print(
                        f"Aligning {rel_idx} ({self.chunk_indices[rel_idx]}) "
                        f"and {chunk_idx} ({self.chunk_indices[chunk_idx]})..."
                    )
                    chunk_data1 = np.load(
                        os.path.join(self.result_unaligned_dir, f"chunk_{rel_idx}.npy"), allow_pickle=True
                    ).item()
                    chunk_data2 = np.load(
                        os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True
                    ).item()

                    point_map1 = chunk_data1["world_points"][-self.overlap :]
                    point_map2 = chunk_data2["world_points"][: self.overlap]
                    conf1 = chunk_data1["world_points_conf"][-self.overlap :]
                    conf2 = chunk_data2["world_points_conf"][: self.overlap]

                    conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

                    mask = None
                    if chunk_data1.get("mask") is not None:
                         mask1 = chunk_data1["mask"][-self.overlap:]
                         mask2 = chunk_data2["mask"][:self.overlap]
                         mask = mask1.squeeze() & mask2.squeeze()

                    s, R, t = weighted_align_point_maps(
                        point_map1, conf1, point_map2, conf2, mask, conf_threshold=conf_threshold, config=self.config
                    )
                    sim3_rel.append((s, R, t))

                    # 3. Create Aligned version for visualization immediately
                    # Compute cumulative Sim3
                    # Recalculate full chain or just append? 
                    # accumulate_sim3_transforms is fast enough for small N
                    sim3_abs_tmp = accumulate_sim3_transforms(sim3_rel)
                    s_abs, R_abs, t_abs = sim3_abs_tmp[chunk_idx - 1]
                    
                    chunk_data2["world_points"] = apply_sim3_direct(chunk_data2["world_points"], s_abs, R_abs, t_abs)
                    aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy")
                    np.save(aligned_path, chunk_data2)
                    
            elif chunk_idx == 0:
                # Chunk 0 is always "aligned" (Identity)
                # Copy to aligned dir for consistency
                shutil.copy2(
                     os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                     os.path.join(self.result_aligned_dir, "chunk_0.npy")
                )

            # 4. Trigger update
            if self.on_chunk_aligned:
                self.on_chunk_aligned(chunk_idx + 1)
        
        # -------------------------------------------------------
        # 3. Loop Closure & Optimization (Global)
        # -------------------------------------------------------
        # Now we have sim3_rel for all chunks.
        sim3_abs = accumulate_sim3_transforms(sim3_rel)
        
        if self.loop_enable:
            print("Running Loop Detection...")
            self.get_loop_pairs()
            
            from loop_utils.sim3utils import process_loop_list
            loop_results = process_loop_list(self.chunk_indices, 
                                             self.loop_list, 
                                             half_window = int(self.config['Model']['loop_chunk_size'] / 2))
            loop_results = remove_duplicates(loop_results)
            print(f"Found valid loop chunk pairs: {len(loop_results)}")

            self.loop_sim3_list = []
            
            for item in loop_results:
                chunk_idx_a = item[0]
                chunk_idx_b = item[2]
                range_a = item[1]
                range_b = item[3]
                
                filename = f"loop_{range_a[0]}_{range_a[1]}_{range_b[0]}_{range_b[1]}.npy"
                loop_path = os.path.join(self.result_loop_dir, filename)
                
                if os.path.exists(loop_path):
                    single_chunk_predictions = np.load(loop_path, allow_pickle=True).item()
                else:
                    print(f"Processing loop pair: {item}")
                    single_chunk_predictions = self.process_single_chunk(range_a, range_2=range_b, is_loop=True)

                chunk_data_a = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"), allow_pickle=True).item()
                chunk_data_b = np.load(os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"), allow_pickle=True).item()
                
                point_map_loop_a = single_chunk_predictions['world_points'][:range_a[1] - range_a[0]]
                conf_loop_a = single_chunk_predictions['world_points_conf'][:range_a[1] - range_a[0]]
                
                chunk_a_rela_begin = range_a[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = chunk_a_rela_begin + range_a[1] - range_a[0]
                
                point_map_a = chunk_data_a['world_points'][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a['world_points_conf'][chunk_a_rela_begin:chunk_a_rela_end]
                
                mask = None
                if single_chunk_predictions.get('mask') is not None:
                     mask_loop = single_chunk_predictions['mask'][:range_a[1] - range_a[0]]
                     mask_a = chunk_data_a['mask'][chunk_a_rela_begin:chunk_a_rela_end]
                     mask = mask_loop.squeeze() & mask_a.squeeze()

                ct = min(np.median(conf_a), np.median(conf_loop_a)) * 0.1
                s_a, R_a, t_a = weighted_align_point_maps(point_map_a, conf_a, point_map_loop_a, conf_loop_a, mask, conf_threshold=ct, config=self.config)

                point_map_loop_b = single_chunk_predictions['world_points'][-(range_b[1] - range_b[0]):]
                conf_loop_b = single_chunk_predictions['world_points_conf'][-(range_b[1] - range_b[0]):]
                
                chunk_b_rela_begin = range_b[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = chunk_b_rela_begin + range_b[1] - range_b[0]
                
                point_map_b = chunk_data_b['world_points'][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b['world_points_conf'][chunk_b_rela_begin:chunk_b_rela_end]
                
                mask = None
                if single_chunk_predictions.get('mask') is not None:
                     mask_loop = single_chunk_predictions['mask'][-(range_b[1] - range_b[0]):]
                     mask_b = chunk_data_b['mask'][chunk_b_rela_begin:chunk_b_rela_end]
                     mask = mask_loop.squeeze() & mask_b.squeeze()

                ct = min(np.median(conf_b), np.median(conf_loop_b)) * 0.1
                s_b, R_b, t_b = weighted_align_point_maps(point_map_b, conf_b, point_map_loop_b, conf_loop_b, mask, conf_threshold=ct, config=self.config)

                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

            if self.loop_sim3_list:
                print("Optimizing Sim3 Pose Graph...")
                optimized_sim3_rel = self.loop_optimizer.optimize(sim3_rel, self.loop_sim3_list)
                sim3_rel = optimized_sim3_rel
                sim3_abs = accumulate_sim3_transforms(sim3_rel)
                print("Optimization done.")

        del self.model
        torch.cuda.empty_cache()

        self.sim3_list = sim3_abs
        self.save_state(sim3_rel, sim3_abs, self.chunk_indices)

        # -------------------------------------------------------
        # 4. Apply & Refresh (Global)
        # -------------------------------------------------------
        print("Applying alignment to ALL chunks (refreshing full sequence)...")
        for chunk_idx in range(len(self.chunk_indices)):
            if chunk_idx == 0:
                s, R, t = 1.0, np.eye(3), np.zeros(3)
                chunk_data = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True
                ).item()
            else:
                s, R, t = sim3_abs[chunk_idx - 1]
                chunk_data = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"), allow_pickle=True
                ).item()
                chunk_data["world_points"] = apply_sim3_direct(chunk_data["world_points"], s, R, t)
            
            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy")
            np.save(aligned_path, chunk_data)

            points = chunk_data["world_points"].reshape(-1, 3)
            colors = (chunk_data["images"].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = chunk_data.get("world_points_conf", np.ones(points.shape[0])).reshape(-1)
            
            ply_path = os.path.join(self.pcd_dir, f"{chunk_idx}_pcd.ply")
            save_confident_pointcloud_batch(
                points=points,
                colors=colors,
                confs=confs,
                output_path=ply_path,
                conf_threshold=np.mean(confs)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

        self.save_camera_poses()

        all_ply_path = os.path.join(self.pcd_dir, "combined_pcd.ply")
        print("Saving all the point clouds to combined_pcd.ply")
        merge_ply_files(self.pcd_dir, all_ply_path)

        if self.on_all_aligned:
            self.on_all_aligned()

        print("Done.")

    def run(self, resume: bool = False):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg")) + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        self.process_long_sequence(resume=resume)

    def save_camera_poses(self):
        """Reconstruct poses/intrinsics from disk chunks (aligned preferred) and write txt/ply."""
        chunk_colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [128, 0, 0],
            [0, 128, 0],
            [0, 0, 128],
            [128, 128, 0],
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        def load_chunk(idx: int):
            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{idx}.npy")
            unaligned_path = os.path.join(self.result_unaligned_dir, f"chunk_{idx}.npy")
            path = aligned_path if os.path.exists(aligned_path) else unaligned_path
            if not os.path.exists(path):
                raise FileNotFoundError(f"chunk file missing: {path}")
            return np.load(path, allow_pickle=True).item()

        for chunk_idx, (start_i, end_i) in enumerate(self.chunk_indices):
            data = load_chunk(chunk_idx)
            extr = data["extrinsic"]
            intr = data["intrinsic"]

            S = np.eye(4)
            if chunk_idx > 0 and len(self.sim3_list) >= chunk_idx:
                s, R, t = self.sim3_list[chunk_idx - 1]
                S[:3, :3] = s * R
                S[:3, 3] = t

            for local_i, idx in enumerate(range(start_i, end_i)):
                w2c = np.eye(4)
                
                # Check if extr is 3x4 or 4x4
                current_extr = extr[local_i]
                if current_extr.shape == (3, 4):
                     w2c[:3, :] = current_extr
                elif current_extr.shape == (4, 4):
                     w2c = current_extr
                else:
                     raise ValueError(f"Unexpected extrinsic shape: {current_extr.shape}")

                c2w = np.linalg.inv(w2c)
                c2w = S @ c2w
                all_poses[idx] = c2w
                
                # Handle potentially None intrinsics safely
                if intr is not None:
                     all_intrinsics[idx] = intr[local_i]
                else:
                     all_intrinsics[idx] = None


        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(" ".join([str(x) for x in flat_pose]) + "\n")

        intrinsics_path = os.path.join(self.output_dir, "intrinsic.txt")
        # Only save intrinsics if at least one is not None
        if any(intr is not None for intr in all_intrinsics):
             with open(intrinsics_path, "w") as f:
                for intrinsic in all_intrinsics:
                    if intrinsic is None:
                        # Skip or write placeholder? Let's skip for now or handle gracefully
                        continue
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    f.write(f"{fx} {fy} {cx} {cy}\n")
             print(f"Camera intrinsics saved to {intrinsics_path}")
        else:
             print("Skipping intrinsic.txt (no intrinsics available)")


        ply_path = os.path.join(self.output_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_poses)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n")

        print(f"Camera poses saved to {poses_path}, {intrinsics_path}, {ply_path}")

    def close(self):
        if not self.delete_temp_files:
            return
        total_space = 0
        print(f"Deleting the temp files under {self.result_unaligned_dir}")
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_aligned_dir}")
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_loop_dir}")
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print("Deleting temp files done.")
        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")


def run_cli():
    import argparse
    import gc

    parser = argparse.ArgumentParser(description="VGGT-Long-Inc")
    parser.add_argument("--image_dir", type=str, required=True, help="Image path")
    parser.add_argument("--config", type=str, required=False, default="./configs/base_config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from existing cache")
    args = parser.parse_args()

    config = load_config(args.config)
    image_dir = args.image_dir
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = "./exps"
    save_dir = os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"The exp will be saved under dir: {save_dir}")
        copy_file(args.config, save_dir)

    if config["Model"]["align_method"] == "numba":
        warmup_numba()

    vggt_long = VGGT_Long_Inc(image_dir, save_dir, config)
    vggt_long.run(resume=args.resume)
    vggt_long.close()

    del vggt_long
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, "pcd/combined_pcd.ply")
    input_dir = os.path.join(save_dir, "pcd")
    # print("Saving all the point clouds")
    # merge_ply_files(input_dir, all_ply_path)
    print("VGGT Long incremental done.")


if __name__ == "__main__":
    run_cli()
