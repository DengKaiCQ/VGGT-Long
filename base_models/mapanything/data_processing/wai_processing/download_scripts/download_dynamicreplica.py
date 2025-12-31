# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/download_scripts/download_dynamicreplica.py

"""
Script to download Dynamic Replica dataset.
It downloads the Dynamic Replica dataset and extracts the archives into a single folder.

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

# URLs for the Dynamic Replica dataset
LINKS = [
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/real/real_000.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_000.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_001.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_002.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_003.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_004.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/valid/valid_005.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_000.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_001.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_002.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_003.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_004.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_005.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_006.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_007.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_008.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_009.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/test/test_010.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_000.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_001.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_002.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_003.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_004.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_005.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_006.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_007.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_008.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_009.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_010.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_011.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_012.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_013.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_014.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_015.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_016.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_017.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_018.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_019.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_020.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_021.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_022.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_023.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_024.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_025.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_026.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_027.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_028.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_029.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_030.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_031.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_032.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_033.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_034.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_035.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_036.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_037.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_038.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_039.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_040.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_041.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_042.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_043.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_044.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_045.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_046.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_047.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_048.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_049.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_050.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_051.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_052.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_053.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_054.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_055.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_056.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_057.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_058.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_059.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_060.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_061.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_062.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_063.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_064.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_065.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_066.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_067.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_068.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_069.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_070.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_071.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_072.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_073.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_074.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_075.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_076.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_077.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_078.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_079.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_080.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_081.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_082.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_083.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_084.zip",
    "https://dl.fbaipublicfiles.com/dynamic_replica_v2/train/train_085.zip",
]


def download_dynamicreplica(target_dir, n_workers):
    """
    Download Dynamic Replica dataset files to the specified directory.

    Args:
        target_dir (str): Target directory to download files to
        n_workers (int): Number of parallel workers for downloading

    Returns:
        bool: True if download was successful, False otherwise
    """
    print(f"Downloading Dynamic Replica dataset to {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    parallel_download(target_dir, LINKS, n_workers=n_workers)

    print("Download complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and process Dynamic Replica dataset"
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
        default=8,
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
        if not download_dynamicreplica(args.target_dir, args.n_workers):
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
