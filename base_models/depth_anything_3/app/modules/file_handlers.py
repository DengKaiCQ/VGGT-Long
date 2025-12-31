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
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/app/modules/file_handlers.py

"""
File handling module for Depth Anything 3 Gradio app.

This module handles file uploads, video processing, and file operations.
"""

import os
import shutil
import time
from datetime import datetime
from typing import List, Optional, Tuple
import cv2
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


class FileHandler:
    """
    Handles file uploads and processing for the Gradio app.
    """

    def __init__(self):
        """Initialize the file handler."""

    def handle_uploads(
        self,
        input_video: Optional[str],
        input_images: Optional[List],
        s_time_interval: float = 10.0,
    ) -> Tuple[str, List[str]]:
        """
        Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
        images or extracted frames from video into it.

        Args:
            input_video: Path to input video file
            input_images: List of input image files
            s_time_interval: Sampling FPS (frames per second) for frame extraction

        Returns:
            Tuple of (target_dir, image_paths)
        """
        start_time = time.time()

        # Get workspace directory from environment variable or use default
        workspace_dir = os.environ.get("DA3_WORKSPACE_DIR", "gradio_workspace")
        if not os.path.exists(workspace_dir):
            os.makedirs(workspace_dir)

        # Create input_images subdirectory
        input_images_dir = os.path.join(workspace_dir, "input_images")
        if not os.path.exists(input_images_dir):
            os.makedirs(input_images_dir)

        # Create a unique folder name within input_images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_dir = os.path.join(input_images_dir, f"session_{timestamp}")
        target_dir_images = os.path.join(target_dir, "images")

        # Clean up if somehow that folder already exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)
        os.makedirs(target_dir_images)

        image_paths = []

        # Handle images
        if input_images is not None:
            image_paths.extend(self._process_images(input_images, target_dir_images))

        # Handle video
        if input_video is not None:
            image_paths.extend(
                self._process_video(input_video, target_dir_images, s_time_interval)
            )

        # Sort final images for gallery
        image_paths = sorted(image_paths)

        end_time = time.time()
        print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
        return target_dir, image_paths

    def _process_images(self, input_images: List, target_dir_images: str) -> List[str]:
        """
        Process uploaded images.

        Args:
            input_images: List of input image files
            target_dir_images: Target directory for images

        Returns:
            List of processed image paths
        """
        image_paths = []

        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data

            # Check if the file is a HEIC image
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in [".heic", ".heif"]:
                # Convert HEIC to JPEG for better gallery compatibility
                try:
                    with Image.open(file_path) as img:
                        # Convert to RGB if necessary (HEIC can have different color modes)
                        if img.mode not in ("RGB", "L"):
                            img = img.convert("RGB")

                        # Create JPEG filename
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        dst_path = os.path.join(target_dir_images, f"{base_name}.jpg")

                        # Save as JPEG with high quality
                        img.save(dst_path, "JPEG", quality=95)
                        image_paths.append(dst_path)
                        print(
                            f"Converted HEIC to JPEG: {os.path.basename(file_path)} -> "
                            f"{os.path.basename(dst_path)}"
                        )
                except Exception as e:
                    print(f"Error converting HEIC file {file_path}: {e}")
                    # Fall back to copying as is
                    dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
                    shutil.copy(file_path, dst_path)
                    image_paths.append(dst_path)
            else:
                # Regular image files - copy as is
                dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
                shutil.copy(file_path, dst_path)
                image_paths.append(dst_path)

        return image_paths

    def _process_video(
        self, input_video: str, target_dir_images: str, s_time_interval: float
    ) -> List[str]:
        """
        Process video file and extract frames.

        Args:
            input_video: Path to input video file
            target_dir_images: Target directory for extracted frames
            s_time_interval: Sampling FPS (frames per second) for frame extraction

        Returns:
            List of extracted frame paths
        """
        image_paths = []

        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / s_time_interval))  # Convert FPS to frame interval

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

        return image_paths

    def update_gallery_on_upload(
        self,
        input_video: Optional[str],
        input_images: Optional[List],
        s_time_interval: float = 10.0,
    ) -> Tuple[Optional[str], Optional[str], Optional[List], Optional[str]]:
        """
        Handle file uploads and update gallery.

        Args:
            input_video: Path to input video file
            input_images: List of input image files
            s_time_interval: Sampling FPS (frames per second) for frame extraction

        Returns:
            Tuple of (reconstruction_output, target_dir, image_paths, log_message)
        """
        if not input_video and not input_images:
            return None, None, None, None

        target_dir, image_paths = self.handle_uploads(input_video, input_images, s_time_interval)
        return (
            None,
            target_dir,
            image_paths,
            "Upload complete. Click 'Reconstruct' to begin 3D processing.",
        )

    def load_example_scene(
        self, scene_name: str, examples_dir: str = "examples"
    ) -> Tuple[Optional[str], Optional[str], Optional[List], str]:
        """
        Load a scene from examples directory.

        Args:
            scene_name: Name of the scene to load
            examples_dir: Path to examples directory

        Returns:
            Tuple of (reconstruction_output, target_dir, image_paths, log_message)
        """
        from depth_anything_3.app.modules.utils import get_scene_info

        scenes = get_scene_info(examples_dir)

        # Find the selected scene
        selected_scene = None
        for scene in scenes:
            if scene["name"] == scene_name:
                selected_scene = scene
                break

        if selected_scene is None:
            return None, None, None, "Scene not found"

        # Use fixed directory name for examples (not timestamp-based)
        workspace_dir = os.environ.get("DA3_WORKSPACE_DIR", "gradio_workspace")
        input_images_dir = os.path.join(workspace_dir, "input_images")
        if not os.path.exists(input_images_dir):
            os.makedirs(input_images_dir)

        # Create a fixed folder name based on scene name
        target_dir = os.path.join(input_images_dir, f"example_{scene_name}")
        target_dir_images = os.path.join(target_dir, "images")

        # Check if already cached (GLB file exists)
        glb_path = os.path.join(target_dir, "scene.glb")
        is_cached = os.path.exists(glb_path)

        # Create directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            os.makedirs(target_dir_images)

        # Copy images if directory is new or empty
        if not os.path.exists(target_dir_images) or len(os.listdir(target_dir_images)) == 0:
            os.makedirs(target_dir_images, exist_ok=True)
            image_paths = []
            for file_path in selected_scene["image_files"]:
                dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
                shutil.copy(file_path, dst_path)
                image_paths.append(dst_path)
        else:
            # Use existing images
            image_paths = sorted(
                [
                    os.path.join(target_dir_images, f)
                    for f in os.listdir(target_dir_images)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"))
                ]
            )

        # Return cached GLB if available
        if is_cached:
            return (
                glb_path,  # Return cached reconstruction
                target_dir,  # Set target directory
                image_paths,  # Set gallery
                f"Loaded cached scene '{scene_name}' with {selected_scene['num_images']} images.",
            )
        else:
            return (
                None,  # No cached reconstruction
                target_dir,  # Set target directory
                image_paths,  # Set gallery
                (
                    f"Loaded scene '{scene_name}' with {selected_scene['num_images']} images. "
                    "Click 'Reconstruct' to begin 3D processing."
                ),
            )
