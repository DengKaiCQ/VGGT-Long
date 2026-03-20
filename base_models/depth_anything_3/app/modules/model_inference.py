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
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/app/modules/model_inference.py

"""
Model inference module for Depth Anything 3 Gradio app.

This module handles all model-related operations including inference,
data processing, and result preparation.
"""

import glob
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.memory import cleanup_cuda_memory
from depth_anything_3.utils.export.glb import export_to_glb
from depth_anything_3.utils.export.gs import export_to_gs_video


class ModelInference:
    """
    Handles model inference and data processing for Depth Anything 3.
    """

    def __init__(self):
        """Initialize the model inference handler."""
        self.model = None

    def initialize_model(self, device: str = "cuda") -> None:
        """
        Initialize the DepthAnything3 model.

        Args:
            device: Device to load the model on
        """
        if self.model is None:
            # Get model directory from environment variable or use default
            model_dir = os.environ.get(
                "DA3_MODEL_DIR", "/dev/shm/da3_models/DA3HF-VITG-METRIC_VITL"
            )
            self.model = DepthAnything3.from_pretrained(model_dir)
            self.model = self.model.to(device)
        else:
            self.model = self.model.to(device)

        self.model.eval()

    def run_inference(
        self,
        target_dir: str,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        process_res_method: str = "upper_bound_resize",
        show_camera: bool = True,
        save_percentage: float = 30.0,
        num_max_points: int = 1_000_000,
        infer_gs: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        gs_trj_mode: str = "extend",
        gs_video_quality: str = "high",
    ) -> Tuple[Any, Dict[int, Dict[str, Any]]]:
        """
        Run DepthAnything3 model inference on images.

        Args:
            target_dir: Directory containing images
            filter_black_bg: Whether to filter black background
            filter_white_bg: Whether to filter white background
            process_res_method: Method for resizing input images
            show_camera: Whether to show camera in 3D view
            save_percentage: Percentage of points to save (0-100)
            num_max_points: Maximum number of points in point cloud
            infer_gs: Whether to infer 3D Gaussian Splatting
            ref_view_strategy: Reference view selection strategy
            gs_trj_mode: Trajectory mode for 3DGS
            gs_video_quality: Video quality for 3DGS

        Returns:
            Tuple of (prediction, processed_data)
        """
        print(f"Processing images from {target_dir}")

        # Device check
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Initialize model if needed
        self.initialize_model(device)

        # Get image paths
        print("Loading images...")
        image_folder_path = os.path.join(target_dir, "images")
        all_image_paths = sorted(glob.glob(os.path.join(image_folder_path, "*")))

        # Filter for image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        all_image_paths = [
            path
            for path in all_image_paths
            if any(path.lower().endswith(ext) for ext in image_extensions)
        ]

        print(f"Found {len(all_image_paths)} images")
        print(f"All image paths: {all_image_paths}")

        # Use sorted image order (reference view will be selected automatically)
        image_paths = all_image_paths
        print(f"Reference view selection strategy: {ref_view_strategy}")

        if len(image_paths) == 0:
            raise ValueError("No images found. Check your upload.")

        # Map UI options to actual method names
        method_mapping = {"high_res": "lower_bound_resize", "low_res": "upper_bound_resize"}
        actual_method = method_mapping.get(process_res_method, "upper_bound_crop")

        # Run model inference
        print(f"Running inference with method: {actual_method}")
        with torch.no_grad():
            prediction = self.model.inference(
                image_paths,
                export_dir=None,
                process_res_method=actual_method,
                infer_gs=infer_gs,
                ref_view_strategy=ref_view_strategy,
            )
        # num_max_points: int = 1_000_000,
        export_to_glb(
            prediction,
            filter_black_bg=filter_black_bg,
            filter_white_bg=filter_white_bg,
            export_dir=target_dir,
            show_cameras=show_camera,
            conf_thresh_percentile=save_percentage,
            num_max_points=int(num_max_points),
        )

        # export to gs video if needed
        if infer_gs:
            mode_mapping = {"extend": "extend", "smooth": "interpolate_smooth"}
            print(f"GS mode: {gs_trj_mode}; Backend mode: {mode_mapping[gs_trj_mode]}")
            export_to_gs_video(
                prediction,
                export_dir=target_dir,
                chunk_size=4,
                trj_mode=mode_mapping.get(gs_trj_mode, "extend"),
                enable_tqdm=True,
                vis_depth="hcat",
                video_quality=gs_video_quality,
            )

        # Save predictions.npz for caching metric depth data
        self._save_predictions_cache(target_dir, prediction)

        # Process results
        processed_data = self._process_results(target_dir, prediction, image_paths)

        # Clean up using centralized memory utilities for consistency with backend
        cleanup_cuda_memory()

        return prediction, processed_data

    def _save_predictions_cache(self, target_dir: str, prediction: Any) -> None:
        """
        Save predictions data to predictions.npz for caching.

        Args:
            target_dir: Directory to save the cache
            prediction: Model prediction object
        """
        try:
            output_file = os.path.join(target_dir, "predictions.npz")

            # Build save dict with prediction data
            save_dict = {}

            # Save processed images if available
            if prediction.processed_images is not None:
                save_dict["images"] = prediction.processed_images

            # Save depth data
            if prediction.depth is not None:
                save_dict["depths"] = np.round(prediction.depth, 6)

            # Save confidence if available
            if prediction.conf is not None:
                save_dict["conf"] = np.round(prediction.conf, 2)

            # Save camera parameters
            if prediction.extrinsics is not None:
                save_dict["extrinsics"] = prediction.extrinsics
            if prediction.intrinsics is not None:
                save_dict["intrinsics"] = prediction.intrinsics

            # Save to file
            np.savez_compressed(output_file, **save_dict)
            print(f"Saved predictions cache to: {output_file}")

        except Exception as e:
            print(f"Warning: Failed to save predictions cache: {e}")

    def _process_results(
        self, target_dir: str, prediction: Any, image_paths: list
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process model results into structured data.

        Args:
            target_dir: Directory containing results
            prediction: Model prediction object
            image_paths: List of input image paths

        Returns:
            Dictionary containing processed data for each view
        """
        processed_data = {}

        # Read generated depth visualization files
        depth_vis_dir = os.path.join(target_dir, "depth_vis")

        if os.path.exists(depth_vis_dir):
            depth_files = sorted(glob.glob(os.path.join(depth_vis_dir, "*.jpg")))
            for i, depth_file in enumerate(depth_files):
                # Use processed images directly from API
                processed_image = None
                if prediction.processed_images is not None and i < len(
                    prediction.processed_images
                ):
                    processed_image = prediction.processed_images[i]

                processed_data[i] = {
                    "depth_image": depth_file,
                    "image": processed_image,
                    "original_image_path": image_paths[i] if i < len(image_paths) else None,
                    "depth": prediction.depth[i] if i < len(prediction.depth) else None,
                    "intrinsics": (
                        prediction.intrinsics[i]
                        if prediction.intrinsics is not None and i < len(prediction.intrinsics)
                        else None
                    ),
                    "mask": None,  # No mask information available
                }

        return processed_data

    # cleanup() removed: call cleanup_cuda_memory() directly where needed.
