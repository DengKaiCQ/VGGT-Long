# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/download_scripts/download_eth3d.py

"""
Script to download ETH3D data for benchmarking
"""

import argparse
import os

import py7zr
import requests
from tqdm import tqdm
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings for insecure download (matching wget --no-check-certificate)
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


def download_file(url, filepath):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True, verify=False)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with (
        open(filepath, "wb") as f,
        tqdm(
            desc=f"Downloading {os.path.basename(filepath)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description="Download ETH3D dataset")
    parser.add_argument("target", type=str, help="Target directory for download")
    args = parser.parse_args()

    target = args.target
    os.makedirs(target, exist_ok=True)

    categories = [
        "courtyard",
        "delivery_area",
        "electro",
        "facade",
        "kicker",
        "meadow",
        "office",
        "pipes",
        "playground",
        "relief",
        "relief_2",
        "terrace",
        "terrains",
    ]
    # Additional available modalities: dslr_raw scan_raw scan_clean scan_eval dslr_occlusion
    datas = ["dslr_jpg", "dslr_undistorted", "dslr_depth"]

    print(f"Downloading ETH3D dataset to {target}")

    for category in categories:
        for data in datas:
            filename = f"{category}_{data}.7z"
            url = f"https://www.eth3d.net/data/{filename}"
            filepath = os.path.join(target, filename)

            # Download file with progress bar
            try:
                download_file(url, filepath)
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {filename}: {e}")
                continue

            # Extract and remove archive
            print(f"Extracting {filename}...")
            try:
                with py7zr.SevenZipFile(filepath, mode="r") as archive:
                    archive.extractall(path=target)
                os.remove(filepath)
            except Exception as e:
                print(f"Failed to extract {filename}: {e}")

    print("Done")


if __name__ == "__main__":
    main()
