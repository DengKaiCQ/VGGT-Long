# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/download_scripts/download_paralleldomain4d.py

"""
Script to download Parallel Domain 4D dataset.
It downloads the Parallel Domain 4D dataset and extracts the archives into a single folder.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from wai_processing.utils.download import (
    extract_tar_archives,
    parallel_download,
)
from wai_processing.utils.parallel import parallel_threads

# URLs for the Parallel Domain 4D dataset
LINKS = [
    "https://tri-ml-public.s3.amazonaws.com/datasets/ParallelDomain-4D.tar",
    "https://gcd.cs.columbia.edu/temporary-public-download/ParallelDomain-4D-OtherModalities.tar",
]


def download_pd4d(target_dir, n_workers):
    """
    Download Parallel Domain 4D dataset files to the specified directory.

    Args:
        target_dir (str): Target directory to download files to
        n_workers (int): Number of parallel workers for downloading

    Returns:
        bool: True if download was successful, False otherwise
    """
    print(f"Downloading Parallel Domain 4D dataset to {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    parallel_download(target_dir, LINKS, n_workers=n_workers)

    print("Download complete!")
    return True


def _merge_single_scene(scene_dir, data_path, human_pose_path, target_path):
    """
    Helper function to merge a single scene directory.

    Args:
        scene_dir (str): Name of the scene directory to process
        data_path (Path): Path to the data directory (or None if doesn't exist)
        human_pose_path (Path): Path to the human_pose directory (or None if doesn't exist)
        target_path (Path): Path to the target paralleldomain4d directory

    Returns:
        bool: True if merge was successful, False otherwise
    """
    try:
        target_scene_path = target_path / scene_dir
        moved_files = 0
        warnings = 0

        # Move contents from data folder
        if data_path:
            source_scene_path = data_path / scene_dir
            if source_scene_path.exists():
                for item in source_scene_path.iterdir():
                    target_item = target_scene_path / item.name
                    if target_item.exists():
                        print(f"Warning: {target_item} already exists, skipping")
                        warnings += 1
                        continue
                    shutil.move(str(item), str(target_item))
                    moved_files += 1

        # Move contents from human_pose folder
        if human_pose_path:
            source_scene_path = human_pose_path / scene_dir
            if source_scene_path.exists():
                for item in source_scene_path.iterdir():
                    target_item = target_scene_path / item.name
                    if target_item.exists():
                        print(f"Warning: {target_item} already exists, skipping")
                        warnings += 1
                        continue
                    shutil.move(str(item), str(target_item))
                    moved_files += 1

        status = f"Merged {scene_dir}: {moved_files} files moved"
        if warnings > 0:
            status += f" ({warnings} warnings)"
        print(status)

        return True

    except Exception as e:
        print(f"Error merging {scene_dir}: {e}")
        return False


def merge_extracted_folders(extract_dir, n_workers):
    """
    Merge the extracted folders into a common paralleldomain4d structure using parallel processing.

    Moves files from:
    - extracted/data/scene_XXXXXX/* -> extracted/paralleldomain4d/scene_XXXXXX/
    - extracted/human_pose/PD_trex_Point_Cache_Final_2_17_2023/scene_XXXXXX/* -> extracted/paralleldomain4d/scene_XXXXXX/

    Args:
        extract_dir (str): Base extraction directory containing 'data' and 'human_pose' folders
        n_workers (int): Number of parallel workers for merging scenes

    Returns:
        bool: True if merge was successful, False otherwise
    """
    print("Starting merge of extracted folders...")

    extract_path = Path(extract_dir)
    data_path = extract_path / "data"
    human_pose_path = (
        extract_path / "human_pose" / "PD_trex_Point_Cache_Final_2_17_2023"
    )
    target_path = extract_path / "paralleldomain4d"

    # Check if source directories exist
    if not data_path.exists():
        print(f"Warning: {data_path} does not exist, skipping data merge")
        data_path = None

    if not human_pose_path.exists():
        print(f"Warning: {human_pose_path} does not exist, skipping human_pose merge")
        human_pose_path = None

    if data_path is None and human_pose_path is None:
        print("Error: Neither source directory exists")
        return False

    # Create target directory
    target_path.mkdir(exist_ok=True)

    # Get all scene directories from both sources
    scene_dirs = set()

    if data_path and data_path.exists():
        scene_dirs.update(
            [
                d.name
                for d in data_path.iterdir()
                if d.is_dir() and d.name.startswith("scene_")
            ]
        )

    if human_pose_path and human_pose_path.exists():
        scene_dirs.update(
            [
                d.name
                for d in human_pose_path.iterdir()
                if d.is_dir() and d.name.startswith("scene_")
            ]
        )

    print(f"Found {len(scene_dirs)} scene directories to merge")

    # Pre-create all target scene directories to avoid race conditions
    print("Creating target scene directories...")
    for scene_dir in sorted(scene_dirs):
        target_scene_path = target_path / scene_dir
        target_scene_path.mkdir(exist_ok=True)

    # Prepare data for parallel processing
    scene_data_list = [
        (scene_dir, data_path, human_pose_path, target_path)
        for scene_dir in sorted(scene_dirs)
    ]

    # Process scene directories in parallel
    print(f"Processing scenes with {n_workers} workers...")
    results = parallel_threads(
        _merge_single_scene,
        scene_data_list,
        workers=n_workers,
        star_args=True,
        desc="Merging scenes",
    )

    # Check if all scenes were processed successfully
    failed_scenes = sum(1 for result in results if not result)
    if failed_scenes > 0:
        print(f"Warning: {failed_scenes} scenes failed to merge")
        return False

    # Clean up empty source directories
    if data_path and data_path.exists():
        try:
            # Remove empty scene directories
            for scene_dir in data_path.iterdir():
                if scene_dir.is_dir() and not any(scene_dir.iterdir()):
                    scene_dir.rmdir()
            # Remove data directory if empty
            if not any(data_path.iterdir()):
                data_path.rmdir()
                print("Removed empty data directory")
        except OSError:
            print("Warning: Could not remove some empty directories in data folder")

    if human_pose_path and human_pose_path.exists():
        try:
            # Remove empty scene directories
            for scene_dir in human_pose_path.iterdir():
                if scene_dir.is_dir() and not any(scene_dir.iterdir()):
                    scene_dir.rmdir()
            # Remove human_pose subdirectory if empty
            if not any(human_pose_path.iterdir()):
                human_pose_path.rmdir()
                # Try to remove parent human_pose directory if empty
                human_pose_parent = human_pose_path.parent
                if not any(human_pose_parent.iterdir()):
                    human_pose_parent.rmdir()
                    print("Removed empty human_pose directory")
        except OSError:
            print(
                "Warning: Could not remove some empty directories in human_pose folder"
            )

    print(f"Merge complete! All scenes merged into {target_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and process Parallel Domain 4D dataset"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Target directory for downloading the dataset",
    )
    parser.add_argument(
        "--extract_dir",
        type=str,
        default=None,
        help="Directory to extract files to (default: <target_dir>/extracted)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=2,
        help="Number of parallel workers for downloading",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["download", "extract", "merge", "all"],
        default=["all"],
        help="Stages to perform: 'download', 'extract', 'merge', or 'all'",
    )

    args = parser.parse_args()

    # Set default directory if not provided
    if args.extract_dir is None:
        args.extract_dir = os.path.join(args.target_dir, "extracted")

    # Determine which operations to perform
    do_download = "download" in args.stages or "all" in args.stages
    do_extract = "extract" in args.stages or "all" in args.stages
    do_merge = "merge" in args.stages or "all" in args.stages

    # Execute the requested operations
    if do_download:
        if not download_pd4d(args.target_dir, args.n_workers):
            print("Download failed. Exiting.")
            return 1

    if do_extract:
        if not extract_tar_archives(args.target_dir, args.extract_dir, args.n_workers):
            print("Extraction failed. Exiting.")
            return 1

    if do_merge:
        if not merge_extracted_folders(args.extract_dir, args.n_workers):
            print("Merge failed. Exiting.")
            return 1

    print("All operations completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
