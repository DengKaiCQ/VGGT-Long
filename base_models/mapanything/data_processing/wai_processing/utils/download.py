# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/utils/download.py

"""
Utility functions for downloads.

References:
Jeff Tan (Carnegie Mellon University)
"""

import os
import signal
import tarfile
import zipfile

from data_processing.wai_processing.utils.parallel import parallel_threads


def extract_zip_archives(target_dir, output_dir=None, n_workers=8):
    """
    Extract the downloaded zip archives in parallel.

    Args:
        target_dir (str): Directory containing the downloaded archives
        output_dir (str, optional): Directory to extract files to. If None, extracts to target_dir/extracted
        n_workers (int, optional): Number of parallel workers for extraction. Defaults to 8.

    Returns:
        bool: True if all extractions were successful, False otherwise
    """
    print("Extracting zip archives...")

    if output_dir is None:
        output_dir = os.path.join(target_dir, "extracted")

    os.makedirs(output_dir, exist_ok=True)

    # Get all zip files in the target directory
    zip_files = [f for f in os.listdir(target_dir) if f.endswith(".zip")]

    if not zip_files:
        print(f"No zip files found in {target_dir}")
        return False

    # Prepare arguments for parallel extraction
    zip_paths = [os.path.join(target_dir, zip_file) for zip_file in zip_files]
    args = [(zip_path, output_dir) for zip_path in zip_paths]

    # Extract archives in parallel
    results = parallel_threads(
        extract_single_zip_archive, args, workers=n_workers, star_args=True, front_num=0
    )

    # Check if all extractions were successful
    if all(results):
        print("Extraction complete!")
        return True
    else:
        print("Some archives failed to extract.")
        return False


def parallel_download(out_folder, links, n_workers=8, no_check_certificate=False):
    """
    Download multiple files in parallel.

    Args:
        out_folder (str): Directory where the files will be saved.
        links (list or dict): Either a list of URLs or a dictionary mapping filenames to URLs.
        n_workers (int, optional): Number of parallel download workers. Defaults to 8.
        no_check_certificate (bool, optional): Whether to skip SSL certificate verification. Defaults to False.

    Raises:
        AssertionError: If links is not a list or dictionary.
    """
    if isinstance(links, list):
        links = {os.path.basename(link): link for link in links}
    else:
        assert isinstance(links, dict)
    args = [
        (out_folder, filename, link, no_check_certificate)
        for filename, link in links.items()
    ]
    parallel_threads(download, args, workers=n_workers, star_args=True, front_num=0)


def extract_single_zip_archive(zip_path, output_dir):
    """
    Extract a single zip archive.

    Args:
        zip_path (str): Path to the zip archive
        output_dir (str): Directory to extract files to

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    zip_file = os.path.basename(zip_path)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted {zip_file}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_file}: {e}")
        return False


def extract_single_tar_archive(tar_path, output_dir):
    """
    Extract a single tar archive.

    Args:
        tar_path (str): Path to the tar archive
        output_dir (str): Directory to extract files to

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    tar_file = os.path.basename(tar_path)

    try:
        with tarfile.open(tar_path, "r:*") as tar_ref:
            tar_ref.extractall(output_dir)
        print(f"Extracted {tar_file}")
        return True
    except Exception as e:
        print(f"Error extracting {tar_file}: {e}")
        return False


def extract_tar_archives(target_dir, output_dir=None, n_workers=8):
    """
    Extract the downloaded tar archives in parallel.

    Args:
        target_dir (str): Directory containing the downloaded archives
        output_dir (str, optional): Directory to extract files to. If None, extracts to target_dir/extracted
        n_workers (int, optional): Number of parallel workers for extraction. Defaults to 8.

    Returns:
        bool: True if all extractions were successful, False otherwise
    """
    print("Extracting tar archives...")

    if output_dir is None:
        output_dir = os.path.join(target_dir, "extracted")

    os.makedirs(output_dir, exist_ok=True)

    # Get all tar files in the target directory
    tar_files = [
        f for f in os.listdir(target_dir) if f.endswith((".tar", ".tar.gz", ".tgz"))
    ]

    if not tar_files:
        print(f"No tar archives found in {target_dir}")
        return False

    # Prepare arguments for parallel extraction
    tar_paths = [os.path.join(target_dir, tar_file) for tar_file in tar_files]
    args = [(tar_path, output_dir) for tar_path in tar_paths]

    # Extract archives in parallel
    results = parallel_threads(
        extract_single_tar_archive, args, workers=n_workers, star_args=True, front_num=0
    )

    # Check if all extractions were successful
    if all(results):
        print("Extraction complete!")
        return True
    else:
        print("Some archives failed to extract.")
        return False


def download(out_folder, filename, link, no_check_certificate=False):
    """
    Download a file, automatically selecting the appropriate download method based on the link.

    Args:
        out_folder (str): Directory where the file will be saved.
        filename (str): Name to save the file as.
        link (str): URL to download the file from.
        no_check_certificate (bool, optional): Whether to skip SSL certificate verification. Defaults to False.
    """
    if "drive.google.com" in link:
        download_gdown(
            out_folder, filename, link, no_check_certificate=no_check_certificate
        )
    else:
        download_wget(
            out_folder, filename, link, no_check_certificate=no_check_certificate
        )


def download_gdown(out_folder, filename, link, no_check_certificate=False):
    """
    Download a file from Google Drive.

    Args:
        out_folder (str): Directory where the file will be saved.
        filename (str): Name to save the file as.
        link (str): Google Drive link to the file.
        no_check_certificate (bool, optional): Whether to skip SSL certificate verification. Defaults to False.

    Raises:
        NotImplementedError: If the Google Drive link format is not recognized.
    """
    filename = f"{out_folder}/{filename}"
    if not os.path.exists(filename):
        # Parse Google drive link
        if "/file/d/" in link and "/view?usp=sharing" in link:
            uid = link.split("/view?usp=sharing")[0].split("/file/d/")[-1]
        elif "/file/d/" in link and "/view?usp=share_link" in link:
            uid = link.split("/view?usp=share_link")[0].split("/file/d/")[-1]
        elif "/open?id=" in link and "&usp=drive_copy" in link:
            uid = link.split("&usp=drive_copy")[0].split("/open?id=")[-1]
        else:
            raise NotImplementedError(f"Cannot parse Google Drive link {link}")

        link = f"https://drive.google.com/uc?id={uid}"

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os_system(
            f"gdown --fuzzy '{link}' -O '{filename}_' && mv '{filename}_' '{filename}'"
        )


def download_wget(out_folder, filename, link, no_check_certificate=False):
    """
    Download a file using wget.

    Args:
        out_folder (str): Directory where the file will be saved.
        filename (str): Name to save the file as.
        link (str): URL to download the file from.
        no_check_certificate (bool, optional): Whether to skip SSL certificate verification. Defaults to False.
    """
    filename = f"{out_folder}/{filename}"
    flags = "--no-check-certificate " if no_check_certificate else ""
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os_system(
            f"wget {flags} -O '{filename}_' '{link}' && mv '{filename}_' '{filename}'"
        )


def os_system(cmd):
    """
    Execute a system command and handle keyboard interrupts.

    Args:
        cmd (str): The command to execute.

    Raises:
        KeyboardInterrupt: If the command is interrupted with SIGINT.
    """
    if signal.SIGINT == os.system(cmd):
        raise KeyboardInterrupt(cmd)
