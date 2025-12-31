# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/download_scripts/download_bmvs.py

"Download the BlendedMVS dataset from the official Github release"

import argparse
import shutil
import subprocess
import zipfile
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

# --- Configuration for BlendedMVS datasets ---

# Details for the original BlendedMVS (low-res) split archives
BLENDEDMVS_LOWRES_CONFIG = {
    "base_url": "https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0",
    "prefix": "BlendedMVS",
    "num_files": 15,
    "combined_name": "BlendedMVS_combined.zip",
}

# Details for the split archives (BlendedMVS+ and BlendedMVS++)
SPLIT_DATASETS_CONFIG = {
    "plus": {
        "base_url": "https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.1",
        "prefix": "BlendedMVS1",
        "num_files": 42,
        "combined_name": "BlendedMVS+_combined.zip",
    },
    "plus_plus": {
        "base_url": "https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.2",
        "prefix": "BlendedMVS2",
        "num_files": 42,
        "combined_name": "BlendedMVS++_combined.zip",
    },
}


def get_all_urls_to_download():
    """Generates a list of all URLs for all BlendedMVS datasets."""
    urls = []

    # Add original BlendedMVS (low-res) split files
    base_url_lowres = BLENDEDMVS_LOWRES_CONFIG["base_url"]
    prefix_lowres = BLENDEDMVS_LOWRES_CONFIG["prefix"]
    for i in range(1, BLENDEDMVS_LOWRES_CONFIG["num_files"] + 1):
        filename = f"{prefix_lowres}.z{i:02d}"
        urls.append(f"{base_url_lowres}/{filename}")
    urls.append(
        f"{base_url_lowres}/{prefix_lowres}.zip"
    )  # Add the final .zip descriptor [2]

    # Add BlendedMVS+ and BlendedMVS++ split files
    for config_name, config in SPLIT_DATASETS_CONFIG.items():
        base_url = config["base_url"]
        prefix = config["prefix"]

        # Add split files (e.g., BlendedMVS1.z01 to BlendedMVS1.z42)
        for i in range(1, config["num_files"] + 1):
            filename = f"{prefix}.z{i:02d}"
            urls.append(f"{base_url}/{filename}")

        # Add the final .zip descriptor (e.g., BlendedMVS1.zip)
        urls.append(f"{base_url}/{prefix}.zip")

    return urls


def download_file(url, save_dir):
    """Downloads a single file with a progress bar, skipping if it exists."""
    local_filename = Path(save_dir) / url.split("/")[-1]
    try:
        # Simple check to skip if file exists and is not empty
        if local_filename.exists() and local_filename.stat().st_size > 0:
            return local_filename

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with (
                open(local_filename, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=local_filename.name,
                    leave=False,
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return local_filename
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {url}: {e}")
        return None


def combine_split_zip(zip_dir, config):
    """Combines split zip files using the 'zip' command-line utility."""
    print(f"\nCombining {config['prefix']} split archives...")
    main_zip_part = zip_dir / f"{config['prefix']}.zip"
    combined_zip_output = zip_dir / config["combined_name"]

    if not main_zip_part.exists():
        print(f"Error: Main zip part {main_zip_part} not found. Cannot combine.")
        return None

    if combined_zip_output.exists() and combined_zip_output.stat().st_size > 0:
        print(f"Skipping combining {config['prefix']}: combined file already exists.")
        return combined_zip_output

    command = [
        "zip",
        "-s-",
        str(main_zip_part.name),
        "-O",
        str(combined_zip_output.name),
    ]

    try:
        subprocess.run(command, cwd=zip_dir, check=True, capture_output=True, text=True)
        print(f"Successfully combined archive: {combined_zip_output}")
        return combined_zip_output
    except FileNotFoundError:
        print(
            "\nError: 'zip' command not found. Please install it to combine archives."
        )
        return None
    except subprocess.CalledProcessError as e:
        print(f"\nError combining archives for {config['prefix']}:\n{e.stderr}")
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred while combining {config['prefix']}: {e}")
        return None


def unzip_file(zip_path, extract_dir):
    """Unzips a single file to the target directory with a progress bar."""
    if not zip_path or not zip_path.exists():
        return f"Skipping unzip: {zip_path} does not exist."

    # All archives are extracted to the same parent directory.
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = zip_ref.infolist()
            # Simple check if already unzipped by checking for one file's existence
            if file_list and (extract_dir / file_list[0].filename).exists():
                print(
                    f"Skipping unzip for {zip_path.name}: content may already be extracted."
                )
                return f"Skipped unzipping {zip_path.name}: assumed already extracted."

            for file in tqdm(file_list, desc=f"Unzipping {zip_path.name}", leave=False):
                zip_ref.extract(file, extract_dir)
        return f"Successfully unzipped {zip_path.name} to {extract_dir}"
    except zipfile.BadZipFile:
        return f"Error: {zip_path.name} is not a valid zip file."
    except Exception as e:
        return f"An error occurred while unzipping {zip_path.name}: {e}"


def flatten_blendedmvs_subdirectory(extract_dir):
    """
    Moves contents from .../BlendedMVS/ up to .../ and removes the empty BlendedMVS folder.
    """
    child_dir = extract_dir / "BlendedMVS"
    if not child_dir.is_dir():
        return  # Nothing to do

    print("\n--- Cleaning up directory structure ---")
    print(f"Moving contents from {child_dir} to {extract_dir}...")

    # Move each item (file or directory) from the child directory to the parent
    for item_path in child_dir.iterdir():
        dest_path = extract_dir / item_path.name
        if dest_path.exists():
            print(
                f"Warning: Destination '{dest_path}' already exists. Skipping move for this item."
            )
            continue
        shutil.move(str(item_path), str(extract_dir))  # Using shutil.move is robust [1]

    # Remove the now-empty child directory
    try:
        child_dir.rmdir()
        print(f"Successfully removed empty directory: {child_dir}")
    except OSError as e:
        print(f"Error removing directory {child_dir}: {e}. It might not be empty.")


def main():
    parser = argparse.ArgumentParser(
        description="Download, combine, and unzip all BlendedMVS datasets into a unified directory."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="The base directory to save and extract the datasets.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers for downloading/unzipping.",
    )
    args = parser.parse_args()

    # 1. Setup directories
    target_dir = Path(args.target_dir)
    zip_dir = target_dir / "blendedmvs_zip"
    extract_dir = target_dir / "blendedmvs"

    zip_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Zip files will be saved in: {zip_dir}")
    print(f"Extracted files will be in: {extract_dir}\n")

    # 2. Download all files in parallel
    print("--- Starting Download ---")
    urls = get_all_urls_to_download()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_url = {
            executor.submit(download_file, url, zip_dir): url for url in urls
        }
        for future in tqdm(
            as_completed(future_to_url),
            total=len(urls),
            desc="Overall Download Progress",
        ):
            future.result()
    print("Download complete.")

    # 3. Combine split archives
    print("\n--- Combining Split Archives ---")
    final_zip_paths = []

    # Combine original BlendedMVS (low-res)
    combined_lowres = combine_split_zip(zip_dir, BLENDEDMVS_LOWRES_CONFIG)
    if combined_lowres:
        final_zip_paths.append(combined_lowres)

    # Combine BlendedMVS+ and BlendedMVS++
    for config_name, config in SPLIT_DATASETS_CONFIG.items():
        combined_file = combine_split_zip(zip_dir, config)
        if combined_file:
            final_zip_paths.append(combined_file)

    if not final_zip_paths:
        print("\nNo main zip files found to unzip. Exiting.")
        return

    # 4. Unzip all final archives into the same directory in parallel
    print("\n--- Starting Unzip ---")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        unique_zip_paths = list(set(final_zip_paths))
        future_to_zip = {
            executor.submit(unzip_file, zip_path, extract_dir): zip_path
            for zip_path in unique_zip_paths
        }
        for future in tqdm(
            as_completed(future_to_zip),
            total=len(unique_zip_paths),
            desc="Overall Unzip Progress",
        ):
            print(future.result())

    flatten_blendedmvs_subdirectory(extract_dir)

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
