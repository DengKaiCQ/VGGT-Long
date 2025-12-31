# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/download_scripts/download_mvs_synth.py

"""
Script to download MVS-Synth dataset.
It downloads the MVS-Synth dataset and extracts the archives into a single folder.

References:
Jeff Tan (Carnegie Mellon University)
"""

import argparse
import os
import sys

from wai_processing.utils.download import (
    extract_tar_archives,
    parallel_download,
)

# URLs for the MVS-Synth dataset
LINKS = [
    # Other source: "https://filebox.ece.vt.edu/~jbhuang/project/deepmvs/mvs-syn/GTAV_1080.tar.gz"
    "https://huggingface.co/datasets/phuang17/MVS-Synth/resolve/main/GTAV_1080.tar.gz",
]


def download_mvs_synth(target_dir, n_workers=1):
    """
    Download MVS-Synth dataset files to the specified directory.

    Args:
        target_dir (str): Target directory to download files to
        n_workers (int): Number of parallel workers for downloading

    Returns:
        bool: True if download was successful, False otherwise
    """
    print(f"Downloading MVS-Synth dataset to {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    parallel_download(target_dir, LINKS, n_workers=n_workers)

    print("Download complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and process MVS-Synth dataset"
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
        default=1,
        help="Number of parallel workers for downloading",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the dataset without extracting or organizing",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract the dataset without downloading or organizing",
    )

    args = parser.parse_args()

    # Set default directory if not provided
    if args.extract_dir is None:
        args.extract_dir = os.path.join(args.target_dir, "extracted")

    # Determine which operations to perform
    do_download = not args.extract_only
    do_extract = not args.download_only

    # Execute the requested operations
    if do_download:
        if not download_mvs_synth(args.target_dir, args.n_workers):
            print("Download failed. Exiting.")
            return 1

    if do_extract:
        if not extract_tar_archives(args.target_dir, args.extract_dir, args.n_workers):
            print("Extraction failed. Exiting.")
            return 1

    print("All operations completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
