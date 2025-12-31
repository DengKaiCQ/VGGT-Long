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
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/app/modules/visualization.py

"""
Visualization module for Depth Anything 3 Gradio app.

This module handles visualization updates, navigation, and measurement functionality.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import gradio as gr
import numpy as np


class VisualizationHandler:
    """
    Handles visualization updates and navigation for the Gradio app.
    """

    def __init__(self):
        """Initialize the visualization handler."""

    def update_view_selectors(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]]
    ) -> Tuple[gr.Dropdown, gr.Dropdown]:
        """
        Update view selector dropdowns based on available views.

        Args:
            processed_data: Processed data dictionary

        Returns:
            Tuple of (depth_view_selector, measure_view_selector)
        """
        if processed_data is None or len(processed_data) == 0:
            choices = ["View 1"]
        else:
            num_views = len(processed_data)
            choices = [f"View {i + 1}" for i in range(num_views)]

        return (
            gr.Dropdown(choices=choices, value=choices[0]),  # depth_view_selector
            gr.Dropdown(choices=choices, value=choices[0]),  # measure_view_selector
        )

    def get_view_data_by_index(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get view data by index, handling bounds.

        Args:
            processed_data: Processed data dictionary
            view_index: Index of the view to get

        Returns:
            View data dictionary or None
        """
        if processed_data is None or len(processed_data) == 0:
            return None

        view_keys = list(processed_data.keys())
        if view_index < 0 or view_index >= len(view_keys):
            view_index = 0

        return processed_data[view_keys[view_index]]

    def update_depth_view(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Optional[str]:
        """
        Update depth view for a specific view index.

        Args:
            processed_data: Processed data dictionary
            view_index: Index of the view to update

        Returns:
            Path to depth visualization image or None
        """
        view_data = self.get_view_data_by_index(processed_data, view_index)
        if view_data is None or view_data.get("depth_image") is None:
            return None

        # Return the depth visualization image directly
        return view_data["depth_image"]

    def navigate_depth_view(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        current_selector_value: str,
        direction: int,
    ) -> Tuple[str, Optional[str]]:
        """
        Navigate depth view (direction: -1 for previous, +1 for next).

        Args:
            processed_data: Processed data dictionary
            current_selector_value: Current selector value
            direction: Direction to navigate (-1 for previous, +1 for next)

        Returns:
            Tuple of (new_selector_value, depth_vis)
        """
        if processed_data is None or len(processed_data) == 0:
            return "View 1", None

        # Parse current view number
        try:
            current_view = int(current_selector_value.split()[1]) - 1
        except:  # noqa
            current_view = 0

        num_views = len(processed_data)
        new_view = (current_view + direction) % num_views

        new_selector_value = f"View {new_view + 1}"
        depth_vis = self.update_depth_view(processed_data, new_view)

        return new_selector_value, depth_vis

    def update_measure_view(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]], view_index: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List]:
        """
        Update measure view for a specific view index.

        Args:
            processed_data: Processed data dictionary
            view_index: Index of the view to update

        Returns:
            Tuple of (measure_image, depth_right_half, measure_points)
        """
        view_data = self.get_view_data_by_index(processed_data, view_index)
        if view_data is None:
            return None, None, []  # image, depth_right_half, measure_points

        # Get the processed (resized) image
        if "image" in view_data and view_data["image"] is not None:
            image = view_data["image"].copy()
        else:
            return None, None, []

        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Extract right half of the depth visualization (pure depth part)
        depth_image_path = view_data.get("depth_image", None)
        depth_right_half = None

        if depth_image_path and os.path.exists(depth_image_path):
            try:
                # Load the combined depth visualization image
                depth_combined = cv2.imread(depth_image_path)
                depth_combined = cv2.cvtColor(depth_combined, cv2.COLOR_BGR2RGB)
                if depth_combined is not None:
                    height, width = depth_combined.shape[:2]
                    # Extract right half (depth visualization part)
                    depth_right_half = depth_combined[:, width // 2 :]
            except Exception as e:
                print(f"Error extracting depth right half: {e}")

        return image, depth_right_half, []

    def navigate_measure_view(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        current_selector_value: str,
        direction: int,
    ) -> Tuple[str, Optional[np.ndarray], Optional[str], List]:
        """
        Navigate measure view (direction: -1 for previous, +1 for next).

        Args:
            processed_data: Processed data dictionary
            current_selector_value: Current selector value
            direction: Direction to navigate (-1 for previous, +1 for next)

        Returns:
            Tuple of (new_selector_value, measure_image, depth_image_path, measure_points)
        """
        if processed_data is None or len(processed_data) == 0:
            return "View 1", None, None, []

        # Parse current view number
        try:
            current_view = int(current_selector_value.split()[1]) - 1
        except:  # noqa
            current_view = 0

        num_views = len(processed_data)
        new_view = (current_view + direction) % num_views

        new_selector_value = f"View {new_view + 1}"
        measure_image, depth_right_half, measure_points = self.update_measure_view(
            processed_data, new_view
        )

        return new_selector_value, measure_image, depth_right_half, measure_points

    def populate_visualization_tabs(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]]
    ) -> Tuple[Optional[str], Optional[np.ndarray], Optional[str], List]:
        """
        Populate the depth and measure tabs with processed data.

        Args:
            processed_data: Processed data dictionary

        Returns:
            Tuple of (depth_vis, measure_img, depth_image_path, measure_points)
        """
        if processed_data is None or len(processed_data) == 0:
            return None, None, None, []

        # Use update function to get depth visualization
        depth_vis = self.update_depth_view(processed_data, 0)
        measure_img, depth_right_half, _ = self.update_measure_view(processed_data, 0)

        return depth_vis, measure_img, depth_right_half, []

    def reset_measure(
        self, processed_data: Optional[Dict[int, Dict[str, Any]]]
    ) -> Tuple[Optional[np.ndarray], List, str]:
        """
        Reset measure points.

        Args:
            processed_data: Processed data dictionary

        Returns:
            Tuple of (image, measure_points, text)
        """
        if processed_data is None or len(processed_data) == 0:
            return None, [], ""

        # Return the first view image
        first_view = list(processed_data.values())[0]
        return first_view["image"], [], ""

    def measure(
        self,
        processed_data: Optional[Dict[int, Dict[str, Any]]],
        measure_points: List,
        current_view_selector: str,
        event: gr.SelectData,
    ) -> List:
        """
        Handle measurement on images.

        Args:
            processed_data: Processed data dictionary
            measure_points: List of current measure points
            current_view_selector: Current view selector value
            event: Gradio select event

        Returns:
            List of [image, depth_right_half, measure_points, text]
        """
        try:
            print(f"Measure function called with selector: {current_view_selector}")

            if processed_data is None or len(processed_data) == 0:
                return [None, [], "No data available"]

            # Use the currently selected view instead of always using the first view
            try:
                current_view_index = int(current_view_selector.split()[1]) - 1
            except:  # noqa
                current_view_index = 0

            print(f"Using view index: {current_view_index}")

            # Get view data safely
            if current_view_index < 0 or current_view_index >= len(processed_data):
                current_view_index = 0

            view_keys = list(processed_data.keys())
            current_view = processed_data[view_keys[current_view_index]]

            if current_view is None:
                return [None, [], "No view data available"]

            point2d = event.index[0], event.index[1]
            print(f"Clicked point: {point2d}")

            measure_points.append(point2d)

            # Get image and depth visualization
            image, depth_right_half, _ = self.update_measure_view(
                processed_data, current_view_index
            )
            if image is None:
                return [None, [], "No image available"]

            image = image.copy()

            # Ensure image is in uint8 format for proper cv2 operations
            try:
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        # Image is in [0, 1] range, convert to [0, 255]
                        image = (image * 255).astype(np.uint8)
                    else:
                        # Image is already in [0, 255] range
                        image = image.astype(np.uint8)
            except Exception as e:
                print(f"Image conversion error: {e}")
                return [None, [], f"Image conversion error: {e}"]

            # Draw circles for points
            try:
                for p in measure_points:
                    if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
                        image = cv2.circle(image, p, radius=5, color=(255, 0, 0), thickness=2)
            except Exception as e:
                print(f"Drawing error: {e}")
                return [None, [], f"Drawing error: {e}"]

            # Get depth information from processed_data
            depth_text = ""
            try:
                for i, p in enumerate(measure_points):
                    if (
                        current_view["depth"] is not None
                        and 0 <= p[1] < current_view["depth"].shape[0]
                        and 0 <= p[0] < current_view["depth"].shape[1]
                    ):
                        d = current_view["depth"][p[1], p[0]]
                        depth_text += f"- **P{i + 1} depth: {d:.2f}m**\n"
                    else:
                        depth_text += f"- **P{i + 1}: Click position ({p[0]}, {p[1]}) - No depth information**\n"  # noqa: E501
            except Exception as e:
                print(f"Depth text error: {e}")
                depth_text = f"Error computing depth: {e}\n"

            if len(measure_points) == 2:
                try:
                    point1, point2 = measure_points
                    # Draw line
                    if (
                        0 <= point1[0] < image.shape[1]
                        and 0 <= point1[1] < image.shape[0]
                        and 0 <= point2[0] < image.shape[1]
                        and 0 <= point2[1] < image.shape[0]
                    ):
                        image = cv2.line(image, point1, point2, color=(255, 0, 0), thickness=2)

                    # Compute 3D distance using depth information and camera intrinsics
                    distance_text = "- **Distance: Unable to calculate 3D distance**"
                    if (
                        current_view["depth"] is not None
                        and 0 <= point1[1] < current_view["depth"].shape[0]
                        and 0 <= point1[0] < current_view["depth"].shape[1]
                        and 0 <= point2[1] < current_view["depth"].shape[0]
                        and 0 <= point2[0] < current_view["depth"].shape[1]
                    ):
                        try:
                            # Get depth values at the two points
                            d1 = current_view["depth"][point1[1], point1[0]]
                            d2 = current_view["depth"][point2[1], point2[0]]

                            # Convert 2D pixel coordinates to 3D world coordinates
                            if current_view["intrinsics"] is not None:
                                # Get camera intrinsics
                                K = current_view["intrinsics"]  # 3x3 intrinsic matrix
                                fx, fy = K[0, 0], K[1, 1]  # focal lengths
                                cx, cy = K[0, 2], K[1, 2]  # principal point

                                # Convert pixel coordinates to normalized camera coordinates
                                # Point 1: (u1, v1) -> (x1, y1, z1)
                                u1, v1 = point1[0], point1[1]
                                x1 = (u1 - cx) * d1 / fx
                                y1 = (v1 - cy) * d1 / fy
                                z1 = d1

                                # Point 2: (u2, v2) -> (x2, y2, z2)
                                u2, v2 = point2[0], point2[1]
                                x2 = (u2 - cx) * d2 / fx
                                y2 = (v2 - cy) * d2 / fy
                                z2 = d2

                                # Calculate 3D Euclidean distance
                                p1_3d = np.array([x1, y1, z1])
                                p2_3d = np.array([x2, y2, z2])
                                distance_3d = np.linalg.norm(p1_3d - p2_3d)

                                distance_text = f"- **Distance: {distance_3d:.2f}m**"
                            else:
                                # Fallback to simplified calculation if no intrinsics
                                pixel_distance = np.sqrt(
                                    (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
                                )
                                avg_depth = (d1 + d2) / 2
                                scale_factor = avg_depth / 1000  # Rough scaling factor
                                estimated_3d_distance = pixel_distance * scale_factor
                                distance_text = f"- **Distance: {estimated_3d_distance:.2f}m (estimated, no intrinsics)**"  # noqa: E501

                        except Exception as e:
                            print(f"Distance computation error: {e}")
                            distance_text = f"- **Distance computation error: {e}**"

                    measure_points = []
                    text = depth_text + distance_text
                    print(f"Measurement complete: {text}")
                    return [image, depth_right_half, measure_points, text]
                except Exception as e:
                    print(f"Final measurement error: {e}")
                    return [None, [], f"Measurement error: {e}"]
            else:
                print(f"Single point measurement: {depth_text}")
                return [image, depth_right_half, measure_points, depth_text]

        except Exception as e:
            print(f"Overall measure function error: {e}")
            return [None, [], f"Measure function error: {e}"]
