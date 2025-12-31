# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/download_scripts/download_unrealstereo4k.py

"""
Script to download UnrealStereo4K dataset.
It downloads the UnrealStereo4K dataset and extracts the archives into a single folder.

References:
Jeff Tan (Carnegie Mellon University)
"""

import argparse
import os
import sys

from wai_processing.utils.download import (
    extract_zip_archives,
    parallel_download,
)

# URLs for the UnrealStereo4K dataset
LINKS = [
    "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00000.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00001.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00002.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00003.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00004.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00005.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00006.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00007.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-projects/smd_nets/UnrealStereo4K_00008.zip",
]


def download_unrealstereo4k(target_dir, n_workers):
    """
    Download UnrealStereo4K dataset files to the specified directory.

    Args:
        target_dir (str): Target directory to download files to
        n_workers (int): Number of parallel workers for downloading
    """
    print(f"Downloading UnrealStereo4K dataset to {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    parallel_download(target_dir, LINKS, n_workers=n_workers)

    print("Download complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and process UnrealStereo4K dataset"
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
        default=9,
        help="Number of parallel workers for downloading & extracting",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["download", "extract", "all"],
        default=["all"],
        help="Stages to perform: 'download', 'extract', or 'all'",
    )

    args = parser.parse_args()

    # Set default directories if not provided
    if args.extract_dir is None:
        args.extract_dir = os.path.join(args.target_dir, "extracted")

    # Determine which operations to perform
    do_download = "download" in args.stages or "all" in args.stages
    do_extract = "extract" in args.stages or "all" in args.stages

    # Execute the requested operations
    if do_download:
        if not download_unrealstereo4k(args.target_dir, args.n_workers):
            print("Download failed. Exiting.")
            return 1

    if do_extract:
        if not extract_zip_archives(args.target_dir, args.extract_dir, args.n_workers):
            print("Extraction failed. Exiting.")
            return 1

    print("All operations completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
