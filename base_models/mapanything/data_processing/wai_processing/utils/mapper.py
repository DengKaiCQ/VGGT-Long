# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/utils/mapper.py

import numpy as np


class DistanceToDepthConverter:
    """
    A class to convert distance maps to depth maps using the pinhole camera model.
    Note: This class assumes that the camera is looking towards the negative z axis and
    this function can deal correctly with non-square pixels, i.e. fx != fy, as this
    occurs in the HMD cameras, and thus also in the xRooms dataset.
    Attributes:
    w (int): The width of the image.
    h (int): The height of the image.
    fl_x (float): The focal length in pixels along the x axis.
    fl_y (float): The focal length in pixels along the y axis.
    cx (float): The principal point in pixels along the x axis.
    cy (float): The principal point in pixels along the y axis.
    Methods:
    distance_to_depth(distance_map): Convert a distance map to a depth map.
    """

    def __init__(self, w, h, fl_x, fl_y, cx, cy, camera_model):
        """
        A class to convert distance maps to depth maps using the pinhole camera model.
        Attributes:
        w (int): The width of the image.
        h (int): The height of the image.
        fl_x (float): The focal length in pixels along the x axis.
        fl_y (float): The focal length in pixels along the y axis.
        cx (float): The principal point in pixels along the x axis.
        cy (float): The principal point in pixels along the y axis.
        Methods:
        distance_to_depth(distance_map): Convert a distance map to a depth map.
        """
        if camera_model != "PINHOLE":
            raise ValueError(
                "Only PINHOLE camera model is supported for dist to depth conversion"
            )
        self.w = w
        self.h = h
        self.fl_x = fl_x
        self.fl_y = fl_y
        self.cx = cx
        self.cy = cy
        self.x_coords = np.arange(w)
        self.y_coords = np.arange(h)
        # Create a meshgrid of x and y coordinates for the target depth map
        # Do this in the constructor to avoid recomputing it every time and we assume
        # the same camera for all frames
        self.x_grid, self.y_grid = np.meshgrid(self.x_coords, self.y_coords)

    def distance_to_depth(self, distance_map):
        """
        Convert distance map to depth map using the pinhole camera model.
        Parameters:
        distance_map (np.ndarray): 2D array representing the distance from the camera to objects in the scene.
        Returns:
        depth_map (np.ndarray): 2D array representing the depth of objects in the scene.
        """
        depth_map = distance_map / np.sqrt(
            (self.x_grid - self.cx) ** 2 / self.fl_x**2
            + (self.y_grid - self.cy) ** 2 / self.fl_y**2
            + 1
        )
        return depth_map
