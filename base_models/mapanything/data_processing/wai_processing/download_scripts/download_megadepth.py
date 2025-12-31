# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/download_scripts/download_megadepth.py

"""
Script to download the MegaDepth dataset.
It downloads the MegaDepth dataset, extracts the archives, and merges the data into a single folder.
Also downloads additional metadata for MegaDepth like pair information.
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request


def download_megadepth(target_dir):
    """
    Download MegaDepth dataset files to the specified directory.

    Args:
        target_dir (str): Target directory to download files to
    """
    print(f"Downloading MegaDepth dataset to {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    # URLs for the dataset files
    urls = {
        "MegaDepth_v1.tar.gz": "https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz",
        "MegaDepth_SfM_v1.tar.xz": "https://www.cs.cornell.edu/projects/megadepth/dataset/MegaDepth_SfM/MegaDepth_SfM_v1.tar.xz",
    }

    for filename, url in urls.items():
        output_path = os.path.join(target_dir, filename)
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded {filename} successfully")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            sys.exit(1)

    print("Download complete!")
    return True


def extract_archives(target_dir):
    """
    Extract the downloaded archives.

    Args:
        target_dir (str): Directory containing the downloaded archives
    """
    print("Extracting archives...")

    # Extract MegaDepth_v1.tar.gz
    v1_path = os.path.join(target_dir, "MegaDepth_v1.tar.gz")
    if os.path.exists(v1_path):
        try:
            subprocess.run(["tar", "-xvf", v1_path, "-C", target_dir], check=True)
            print("Extracted MegaDepth_v1.tar.gz")
        except Exception as e:
            print(f"Error extracting MegaDepth_v1.tar.gz: {e}")
            return False
    else:
        print(f"Warning: {v1_path} not found")
        return False

    # Extract MegaDepth_SfM_v1.tar.xz
    sfm_path = os.path.join(target_dir, "MegaDepth_SfM_v1.tar.xz")
    if os.path.exists(sfm_path):
        try:
            subprocess.run(["tar", "-xvf", sfm_path, "-C", target_dir], check=True)
            print("Extracted MegaDepth_SfM_v1.tar.xz")
        except Exception as e:
            print(f"Error extracting MegaDepth_SfM_v1.tar.xz: {e}")
            return False
    else:
        print(f"Warning: {sfm_path} not found")
        return False

    print("Extraction complete!")
    return True


def merge_megadepth(root_dir):
    """
    Merge the Dense Image and SfM folders of MegaDepth.

    Args:
        root_dir (str): Root directory containing MegaDepth_v1 and MegaDepth_v1_SfM folders
    """
    print("Merging MegaDepth data...")

    mega_depth_v1 = os.path.join(root_dir, "megadepth_v1")
    mega_depth_v1_sfm = os.path.join(root_dir, "megadepth_v1_SfM")
    combined_folder = os.path.join(root_dir, "megadepth")

    # Check if source directories exist
    if not os.path.exists(mega_depth_v1):
        print(f"Error: {mega_depth_v1} directory not found")
        return False

    if not os.path.exists(mega_depth_v1_sfm):
        print(f"Error: {mega_depth_v1_sfm} directory not found")
        return False

    # Create the combined folder if it doesn't exist
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)

    # Get the list of scene subfolders in MegaDepth_v1 that also exist in MegaDepth_v1_SfM
    scene_subfolders = [
        folder
        for folder in os.listdir(mega_depth_v1)
        if os.path.isdir(os.path.join(mega_depth_v1_sfm, folder))
    ]

    # Iterate through each scene subfolder
    for scene in scene_subfolders:
        print(f"Processing scene {scene}...")
        # Create the scene subfolder in the combined folder if it doesn't exist
        combined_scene_folder = os.path.join(combined_folder, scene)
        if not os.path.exists(combined_scene_folder):
            os.makedirs(combined_scene_folder)

        # Move files from MegaDepth_v1 to the combined folder
        mega_depth_v1_scene_folder = os.path.join(mega_depth_v1, scene)
        for data_subfolder in os.listdir(mega_depth_v1_scene_folder):
            source_path = os.path.join(mega_depth_v1_scene_folder, data_subfolder)
            dest_path = os.path.join(combined_scene_folder, data_subfolder)
            shutil.move(source_path, dest_path)

        # Move files from MegaDepth_v1_SfM to the combined folder
        mega_depth_v1_sfm_scene_folder = os.path.join(mega_depth_v1_sfm, scene)
        for data_subfolder in os.listdir(mega_depth_v1_sfm_scene_folder):
            source_path = os.path.join(mega_depth_v1_sfm_scene_folder, data_subfolder)
            dest_path = os.path.join(combined_scene_folder, data_subfolder)
            shutil.move(source_path, dest_path)

    # Delete the original folders
    shutil.rmtree(mega_depth_v1)
    shutil.rmtree(mega_depth_v1_sfm)

    print("Merge complete!")
    return True


def download_megadepth_pairs(target_dir):
    """
    Download MegaDepth pairs file to the combined megadepth folder.

    Args:
        target_dir (str): Root directory containing the combined megadepth folder
    """
    print("Downloading MegaDepth pairs file...")

    combined_folder = os.path.join(target_dir, "megadepth")
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)

    pairs_url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/megadepth_pairs.npz"
    output_path = os.path.join(combined_folder, "megadepth_pairs.npz")

    try:
        print(f"Downloading megadepth_pairs.npz to {output_path}...")
        urllib.request.urlretrieve(pairs_url, output_path)
        print("Downloaded megadepth_pairs.npz successfully")
        return True
    except Exception as e:
        print(f"Error downloading megadepth_pairs.npz: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and process MegaDepth dataset"
    )
    parser.add_argument(
        "target_dir",
        type=str,
        help="Target directory for downloading and processing the dataset",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the dataset without extracting or merging",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge the dataset without downloading or extracting",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction of archives (use if already extracted)",
    )

    args = parser.parse_args()

    # Determine which operations to perform
    do_download = not args.merge_only
    do_extract = not (args.download_only or args.skip_extract or args.merge_only)
    do_merge = not args.download_only

    # Execute the requested operations
    if do_download:
        if not download_megadepth(args.target_dir):
            print("Download failed. Exiting.")
            return 1

        if not download_megadepth_pairs(args.target_dir):
            print("Downloading MegaDepth pairs file failed. Exiting.")
            return 1

    if do_extract:
        if not extract_archives(args.target_dir):
            print("Extraction failed. Exiting.")
            return 1

    if do_merge:
        if not merge_megadepth(args.target_dir):
            print("Merge failed. Exiting.")
            return 1

    print("All operations completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
