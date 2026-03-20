# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/data/bonn_preprocessing.py

import os
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm


def process_bonn(data_path):
    """Process Bonn dataset: extract 110 frames (index 30-140) for rgb, depth, and groundtruth."""
    dataset_dir = os.path.join(data_path, "rgbd_bonn_dataset")
    dirs = sorted(glob.glob(os.path.join(dataset_dir, "*/")))

    for scene_dir in tqdm(dirs, desc="Processing Bonn scenes"):
        # Copy RGB frames 30-140
        rgb_frames = sorted(glob.glob(os.path.join(scene_dir, "rgb", "*.png")))
        rgb_subset = rgb_frames[30:140]
        rgb_out = os.path.join(scene_dir, "rgb_110")
        os.makedirs(rgb_out, exist_ok=True)
        for frame in rgb_subset:
            shutil.copy(frame, rgb_out)

        # Copy depth frames 30-140
        depth_frames = sorted(glob.glob(os.path.join(scene_dir, "depth", "*.png")))
        depth_subset = depth_frames[30:140]
        depth_out = os.path.join(scene_dir, "depth_110")
        os.makedirs(depth_out, exist_ok=True)
        for frame in depth_subset:
            shutil.copy(frame, depth_out)

        # Slice groundtruth poses 30-140
        gt_path = os.path.join(scene_dir, "groundtruth.txt")
        if os.path.exists(gt_path):
            gt = np.loadtxt(gt_path)
            gt_subset = gt[30:140]
            np.savetxt(os.path.join(scene_dir, "groundtruth_110.txt"), gt_subset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the Bonn data folder")
    args = parser.parse_args()
    process_bonn(args.data_path)
