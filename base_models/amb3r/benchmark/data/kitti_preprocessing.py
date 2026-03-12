# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/data/kitti_preprocessing.py

import os
import glob
import shutil
import argparse
from tqdm import tqdm


def process_kitti(data_path):
    """Process KITTI dataset: gather depth and image files into a unified directory structure."""
    val_dir = os.path.join(data_path, "val")
    out_base = os.path.join(data_path, "depth_selection", "val_selection_cropped")

    depth_dirs = sorted(glob.glob(os.path.join(val_dir, "*", "proj_depth", "groundtruth", "image_02")))

    for depth_dir in tqdm(depth_dirs, desc="Processing KITTI scenes"):
        drive_name = depth_dir.split(os.sep)[-4]  # e.g. 2011_09_26_drive_0002_sync
        date_prefix = "_".join(drive_name.split("_")[:3])  # e.g. 2011_09_26

        gt_depth_out = os.path.join(out_base, "groundtruth_depth_gathered", f"{drive_name}_02")
        image_out = os.path.join(out_base, "image_gathered", f"{drive_name}_02")
        os.makedirs(gt_depth_out, exist_ok=True)
        os.makedirs(image_out, exist_ok=True)

        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))[:110]

        for depth_file in depth_files:
            filename = os.path.basename(depth_file)

            # Copy depth file
            shutil.copy(depth_file, os.path.join(gt_depth_out, filename))

            # Find and copy corresponding RGB image
            image_file = os.path.join(
                data_path, date_prefix, drive_name, "image_02", "data", filename
            )
            if os.path.exists(image_file):
                shutil.copy(image_file, os.path.join(image_out, filename))
            else:
                print(f"Warning: Image file not found: {image_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the KITTI data folder")
    args = parser.parse_args()
    process_kitti(args.data_path)
