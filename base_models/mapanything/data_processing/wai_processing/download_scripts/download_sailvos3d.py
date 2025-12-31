# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/map-anything/blob/main/data_processing/wai_processing/download_scripts/download_sailvos3d.py

"""
Script to download SAIL-VOS 3D dataset.
It downloads the SAIL-VOS 3D dataset and extracts the archives following a specific extraction process.

References:
Jeff Tan (Carnegie Mellon University)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from wai_processing.utils.download import parallel_download
from wai_processing.utils.parallel import parallel_threads

# URLs for the SAIL-VOS 3D dataset
LINKS = {
    # Train: RGB frames, meshes, instance segmentation, binvox file, MSCOCO style JSON annotation, camera params
    "sailvos3d_train_images.tar.001": "https://uofi.box.com/shared/static/3eripmzmomw5702ni3twfrk0l8743fxs.001",
    "sailvos3d_train_images.tar.002": "https://uofi.box.com/shared/static/8jib67fxo8lwzqdytxb1h1ort9uxsn2a.002",
    "sailvos3d_train_images.tar.003": "https://uofi.box.com/shared/static/qix2bb0q03qx1rw982f3uuaoeqdbvclk.003",
    "sailvos3d_train_images.tar.004": "https://uofi.box.com/shared/static/pwt8k9d4s5rvakt8n9on6nid7aua6qp3.004",
    "sailvos3d_train_images.tar.005": "https://uofi.box.com/shared/static/20mx6rotndrw13bqlwrd9iz0ymr838pw.005",
    "sailvos3d_train_images.tar.006": "https://uofi.box.com/shared/static/fy922rc89rqc8jxrnymou4ef26rmyyux.006",
    "sailvos3d_train_images.tar.007": "https://uofi.box.com/shared/static/j3wk6g882kkkc266011amcx7m4wc7qz7.007",
    "sailvos3d_train_images.tar.008": "https://uofi.box.com/shared/static/pyace6q4r9azoo5s64omfq6i0q7uwbu0.008",
    "sailvos3d_train_depth.tar.001": "https://uofi.box.com/shared/static/ro2dbtfbmb9lg1104v4rrvllzojr33tt.001",
    "sailvos3d_train_depth.tar.002": "https://uofi.box.com/shared/static/69fd5zbce3a3p5va0js4r3g1easnf90u.002",
    "sailvos3d_train_depth.tar.003": "https://uofi.box.com/shared/static/1vvnh1fmqg1j92709hho86cwziserjm9.003",
    "sailvos3d_train_depth.tar.004": "https://uofi.box.com/shared/static/z4g1mbpn656jwb8vyewmjgo2s4kmgdx2.004",
    "sailvos3d_train_depth.tar.005": "https://uofi.box.com/shared/static/vedj0h131a2bo27vy1teg8xrpm0r94nk.005",
    "sailvos3d_train_depth.tar.006": "https://uofi.box.com/shared/static/s8kplxffgbl20nq6811lx0vu50tmm6mu.006",
    "sailvos3d_train_depth.tar.007": "https://uofi.box.com/shared/static/57lp89cvkef32z2e6qgkmeakd2ib83qk.007",
    "sailvos3d_train_depth.tar.008": "https://uofi.box.com/shared/static/5apa94i6mycd2f25vwja7zrrspnxxtub.008",
    "sailvos3d_train_depth.tar.009": "https://uofi.box.com/shared/static/gn01iqa1lsc7varcwwpckxg264x9rnub.009",
    "sailvos3d_train_depth.tar.010": "https://uofi.box.com/shared/static/kh7uad59z1mgpiimyd5zoju53tq6qfoa.010",
    "sailvos3d_train_depth.tar.011": "https://uofi.box.com/shared/static/ytbnj7ipwo8mcswpyc4h7ufl5zeo9zj8.011",
    "sailvos3d_train_depth.tar.012": "https://uofi.box.com/shared/static/bko72b2ozj9wi2mc95n8x6g8dqnjt5j6.012",
    "sailvos3d_train_depth.tar.013": "https://uofi.box.com/shared/static/jux284t3hrqo6lz56f4km4yrxvv48tn9.013",
    "sailvos3d_train_depth.tar.014": "https://uofi.box.com/shared/static/yhaqa0fu947cztgv631he7yg1614ya08.014",
    "sailvos3d_train_depth.tar.015": "https://uofi.box.com/shared/static/72b53cf64tinuyiedyy6pemaxei2q9rw.015",
    "sailvos3d_train_mesh.tar.001": "https://uofi.box.com/shared/static/m8efy5ig99ld0qw4de83ecb2vlpqfk1x.001",
    "sailvos3d_train_mesh.tar.002": "https://uofi.box.com/shared/static/jjfvsdk09xie8mh8dgudl8tpj2fgot2h.002",
    "sailvos3d_train_mesh.tar.003": "https://uofi.box.com/shared/static/3rlv1nmljar7nzn05qbyu8446pefv2dg.003",
    "sailvos3d_train_mesh.tar.004": "https://uofi.box.com/shared/static/ik94tms25zru0ppwoz2suggvrs08pewx.004",
    "sailvos3d_train_mesh.tar.005": "https://uofi.box.com/shared/static/5xm7qqoe3a543kd8vzlu05kn4vi80tut.005",
    "sailvos3d_train_mesh.tar.006": "https://uofi.box.com/shared/static/tiyzm7hd17u5gou00qvruu64znl6b7ni.006",
    "sailvos3d_train_visible.zip": "https://uofi.box.com/shared/static/3247j2yt5mfvfh69ka80c5apj9bhacrk.zip",
    "sailvos3d_train_voxel.zip": "https://uofi.box.com/shared/static/y51av4iow9aoxhl3yoas8drpqb25s1mk.zip",
    "sailvos3d_train_camera.zip": "https://uofi.box.com/shared/static/xz0j5rza2odwqny05kmw4g5yf1hxte7t.zip",
    "sailvos3d_train_rage_matrices.zip": "https://uofi.box.com/shared/static/gouw67icrievh139w80nhetzeojoi1pp.zip",
    "sailvos3d_train_annot_24.json": "https://uofi.box.com/shared/static/k46rj1zwi7brtsq55gufsrpb50lvypjt.json",
    "sailvos3d_train_split.txt": "https://uofi.box.com/shared/static/cz4lqofgbq5dqq5qrhqt9vxeme1ac0om.txt",
    # Val: RGB frames, meshes, instance segmentation, binvox file, MSCOCO style JSON annotation, camera params
    "sailvos3d_val_images.tar.001": "https://uofi.box.com/shared/static/f22dfmi9wfslqty71yjvrqxg3rbdznbe.001",
    "sailvos3d_val_images.tar.002": "https://uofi.box.com/shared/static/aifckiwh01dkgs9ty5rlfk9oyjf2bf2w.002",
    "sailvos3d_val_depth.tar.001": "https://uofi.box.com/shared/static/f63ma0q8aa0xaghjpsv1n4sfz0kvlnzo.001",
    "sailvos3d_val_depth.tar.002": "https://uofi.box.com/shared/static/i41albg2qvaaholpsigyn5m0wzhp79q6.002",
    "sailvos3d_val_depth.tar.003": "https://uofi.box.com/shared/static/1idnnpqfbwnwyt3hqfmk1pusg81zbn0q.003",
    "sailvos3d_val_depth.tar.004": "https://uofi.box.com/shared/static/c9k5o0y2q8wtl5bkppdnz56aufdboe3p.004",
    "sailvos3d_val_depth.tar.005": "https://uofi.box.com/shared/static/i0kzp2ubn9bpqqqs01457bpx3u4d3zu9.005",
    "sailvos3d_val_mesh.tar.001": "https://uofi.box.com/shared/static/a404zcz7ntrukzxo3hmp90bvccmzkpgy.001",
    "sailvos3d_val_mesh.tar.002": "https://uofi.box.com/shared/static/i1btueb5ncj2q2uedhtny47i5m7baddl.002",
    "sailvos3d_val_mesh.tar.003": "https://uofi.box.com/shared/static/675u55ai5ra0qnl1zvetxb6ii9czj2z5.003",
    "sailvos3d_val_visible.zip": "https://uofi.box.com/shared/static/bkps8e53r711qynsinofnnwgzoexikh2.zip",
    "sailvos3d_val_voxel.zip": "https://uofi.box.com/shared/static/euomlke8gu3ifuog9mx0roibab48ce8g.zip",
    "sailvos3d_val_camera.zip": "https://uofi.box.com/shared/static/a07b3biw6c8plc7x7tb9q9u4uyiiohg2.zip",
    "sailvos3d_val_rage_matrices.zip": "https://uofi.box.com/shared/static/6vew1eje42rwwm13m3y4q8ishbdsiib6.zip",
    "sailvos3d_val_annot_24.json": "https://uofi.box.com/shared/static/zqn1vnxwdp6g54mmzoz977cpjux76the.json",
    "sailvos3d_val_split.txt": "https://uofi.box.com/shared/static/a8vq5asaykxph37ezru7scxjm819ya42.txt",
}


def download_sailvos3d(target_dir, n_workers):
    """
    Download SAIL-VOS 3D dataset files to the specified directory.

    Args:
        target_dir (str): Target directory to download files to
        n_workers (int): Number of parallel workers for downloading

    Returns:
        bool: True if download was successful
    """
    print(f"Downloading SAIL-VOS 3D dataset to {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    parallel_download(target_dir, LINKS, n_workers=n_workers)

    print("Download complete!")
    return True


def run_command(cmd):
    """
    Run a shell command and return success status.

    Args:
        cmd (str): Command to run

    Returns:
        bool: True if command succeeded, False otherwise
    """
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd}\nError: {e}")
        return False


def untar_split_files(prefix, num_parts):
    """
    Concatenate and extract split tar files.

    Args:
        prefix (str): Prefix of the tar files (e.g., "sailvos3d_train_images.tar")
        num_parts (int): Number of parts

    Returns:
        bool: True if extraction was successful
    """
    if all(os.path.exists(f"{prefix}.{i:03d}") for i in range(1, num_parts + 1)):
        print(f"Extracting {prefix}.*...")
        return run_command(f"cat {prefix}.* | tar xvf -")
    return True  # Skip if files don't exist


def untar_sailvos3d(target_dir, n_workers):
    """
    Stage 1: Untar the split tar files to get zip files.

    Args:
        target_dir (str): Directory containing the downloaded files
        n_workers (int): Number of parallel workers for extraction

    Returns:
        bool: True if untarring was successful
    """
    print("Stage 1: Untarring split tar files in parallel...")

    # Change to the target directory to perform extraction
    original_dir = os.getcwd()
    os.chdir(target_dir)

    try:
        # Define the tar file sets to extract
        tar_tasks = [
            ("sailvos3d_train_images.tar", 8),
            ("sailvos3d_train_depth.tar", 15),
            ("sailvos3d_train_mesh.tar", 6),
            ("sailvos3d_val_images.tar", 2),
            ("sailvos3d_val_depth.tar", 5),
            ("sailvos3d_val_mesh.tar", 3),
        ]

        # Run untarring in parallel
        args = [(prefix, num_parts) for prefix, num_parts in tar_tasks]
        results = parallel_threads(
            untar_split_files, args, workers=n_workers, star_args=True, front_num=0
        )

        if not all(results):
            print("Some tar extraction tasks failed.")
            return False

        print("Untar stage complete!")
        return True

    except Exception as e:
        print(f"Untar stage failed: {e}")
        return False
    finally:
        # Return to the original directory
        os.chdir(original_dir)


def extract_zips_sailvos3d(target_dir, extract_dir, n_workers):
    """
    Stage 2: Extract zip files to the final directory structure.

    Args:
        target_dir (str): Directory containing the zip files
        extract_dir (str): Directory to extract files to
        n_workers (int): Number of parallel workers for extraction

    Returns:
        bool: True if zip extraction was successful
    """
    print("Stage 2: Extracting zip files to final directory structure...")

    # Create the sailvos3d directory in the extract_dir
    sailvos3d_dir = os.path.join(extract_dir, "sailvos3d")
    os.makedirs(sailvos3d_dir, exist_ok=True)

    # Change to the target directory to perform extraction
    original_dir = os.getcwd()
    os.chdir(target_dir)

    try:
        # Extract zip files directly to the sailvos3d directory in parallel
        parallel_zips = [
            "sailvos3d_train_images.zip",
            "sailvos3d_train_depth.zip",
            "sailvos3d_train_visible.zip",
            "sailvos3d_train_voxel.zip",
            "sailvos3d_train_camera.zip",
            "sailvos3d_train_rage_matrices.zip",
            "sailvos3d_val_images.zip",
            "sailvos3d_val_depth.zip",
            "sailvos3d_val_visible.zip",
            "sailvos3d_val_voxel.zip",
            "sailvos3d_val_camera.zip",
            "sailvos3d_val_rage_matrices.zip",
        ]

        # Filter to only existing zip files
        existing_zips = [
            zip_file for zip_file in parallel_zips if os.path.exists(zip_file)
        ]

        if existing_zips:
            print(f"Extracting {len(existing_zips)} zip files in parallel...")

            def extract_zip(zip_file):
                print(f"Extracting {zip_file}...")
                return run_command(f"unzip -q {zip_file} -d {extract_dir}/sailvos3d")

            # Run zip extraction in parallel
            results = parallel_threads(
                extract_zip, existing_zips, workers=n_workers, front_num=0
            )

            if not all(results):
                print("Some zip extraction tasks failed.")
                return False

        print("Zip extraction stage complete!")
        return True

    except Exception as e:
        print(f"Zip extraction stage failed: {e}")
        return False
    finally:
        # Return to the original directory
        os.chdir(original_dir)


def organize_files_sailvos3d(target_dir, extract_dir):
    """
    Stage 3: Organize JSON and text files to their final locations.

    Args:
        target_dir (str): Directory containing the files to organize
        extract_dir (str): Directory where files should be organized

    Returns:
        bool: True if file organization was successful
    """
    print("Stage 5: Organizing JSON and text files...")

    # Create the sailvos3d directory in the extract_dir
    sailvos3d_dir = os.path.join(extract_dir, "sailvos3d")
    os.makedirs(sailvos3d_dir, exist_ok=True)

    # Change to the target directory to perform file operations
    original_dir = os.getcwd()
    os.chdir(target_dir)

    try:
        # Move JSON and txt files
        for file in [
            "sailvos3d_train_annot_24.json",
            "sailvos3d_train_split.txt",
            "sailvos3d_val_annot_24.json",
            "sailvos3d_val_split.txt",
        ]:
            if os.path.exists(file):
                if not run_command(f"mv {file} {extract_dir}/sailvos3d/"):
                    return False

        print("File organization stage complete!")
        return True

    except Exception as e:
        print(f"File organization stage failed: {e}")
        return False
    finally:
        # Return to the original directory
        os.chdir(original_dir)


def extract_mesh_zips_sailvos3d(target_dir):
    """
    Stage 4: Extract mesh zip files.

    Args:
        target_dir (str): Directory containing the mesh zip files

    Returns:
        bool: True if mesh zip extraction was successful
    """
    print("Stage 3: Extracting mesh zip files...")

    # Change to the target directory to perform extraction
    original_dir = os.getcwd()
    os.chdir(target_dir)

    try:
        # Extract val meshes
        if os.path.exists("sailvos3d_val_mesh.zip"):
            if not run_command(
                "unzip -q sailvos3d_val_mesh.zip 'sailvos3d_val_mesh/*'"
            ):
                return False

        # Extract train meshes
        if os.path.exists("sailvos3d_train_mesh.zip"):
            if not run_command(
                "unzip -q sailvos3d_train_mesh.zip 'sailvos3d_train_mesh/*'"
            ):
                return False

        print("Mesh zip extraction stage complete!")
        return True

    except Exception as e:
        print(f"Mesh zip extraction stage failed: {e}")
        return False
    finally:
        # Return to the original directory
        os.chdir(original_dir)


def _merge_directory_contents(source_dir, target_dir):
    """
    Recursively merge contents from source directory to target directory.

    Args:
        source_dir (Path): Source directory to merge from
        target_dir (Path): Target directory to merge into

    Returns:
        tuple: (moved_files_count, warnings_count)
    """
    moved_files = 0
    warnings = 0

    for item in source_dir.iterdir():
        target_item = target_dir / item.name

        if item.is_file():
            # Handle file
            if target_item.exists():
                print(f"Warning: File {target_item} already exists, skipping")
                warnings += 1
                continue
            try:
                # Ensure target directory exists
                target_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(target_item))
                moved_files += 1
            except Exception as e:
                print(f"Error moving file {item}: {e}")
                raise

        elif item.is_dir():
            # Handle directory - merge contents recursively
            if not target_item.exists():
                # Target directory doesn't exist, can move entire directory
                try:
                    target_item.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(item), str(target_item))
                    moved_files += 1
                except Exception as e:
                    print(f"Error moving directory {item}: {e}")
                    raise
            else:
                # Target directory exists, need to merge contents
                print(f"Merging contents of {item.name} into existing directory")
                sub_moved, sub_warnings = _merge_directory_contents(item, target_item)
                moved_files += sub_moved
                warnings += sub_warnings

                # Remove source directory if it's now empty
                try:
                    if not any(item.iterdir()):
                        item.rmdir()
                except OSError:
                    pass  # Directory not empty or other issue, ignore

    return moved_files, warnings


def move_meshes_sailvos3d(target_dir, extract_dir):
    """
    Stage 5: Move extracted mesh files to their final locations.
    Uses robust recursive merging to handle nested directory structures.

    Args:
        target_dir (str): Directory containing the extracted mesh files
        extract_dir (str): Directory where mesh files should be moved

    Returns:
        bool: True if mesh file moving was successful
    """
    print("Stage 4: Moving extracted mesh files...")

    # Create the sailvos3d directory in the extract_dir
    sailvos3d_dir = Path(extract_dir) / "sailvos3d"
    sailvos3d_dir.mkdir(parents=True, exist_ok=True)

    # Change to the target directory to perform file operations
    original_dir = os.getcwd()
    os.chdir(target_dir)

    try:
        total_moved_files = 0
        total_warnings = 0

        # Process train meshes
        train_mesh_dir = Path("sailvos3d_train_mesh")
        if train_mesh_dir.exists():
            print("Merging train mesh files...")
            moved_files, warnings = _merge_directory_contents(
                train_mesh_dir, sailvos3d_dir
            )
            total_moved_files += moved_files
            total_warnings += warnings

            # Clean up empty source directory
            try:
                if not any(train_mesh_dir.iterdir()):
                    train_mesh_dir.rmdir()
                    print("Removed empty train mesh directory")
            except OSError:
                pass  # Directory not empty or other issue

        # Process val meshes
        val_mesh_dir = Path("sailvos3d_val_mesh")
        if val_mesh_dir.exists():
            print("Merging val mesh files...")
            moved_files, warnings = _merge_directory_contents(
                val_mesh_dir, sailvos3d_dir
            )
            total_moved_files += moved_files
            total_warnings += warnings

            # Clean up empty source directory
            try:
                if not any(val_mesh_dir.iterdir()):
                    val_mesh_dir.rmdir()
                    print("Removed empty val mesh directory")
            except OSError:
                pass  # Directory not empty or other issue

        status = f"Mesh file moving complete: {total_moved_files} files moved"
        if total_warnings > 0:
            status += f" ({total_warnings} warnings)"
        print(status)
        return True

    except Exception as e:
        print(f"Mesh file moving stage failed: {e}")
        return False
    finally:
        # Return to the original directory
        os.chdir(original_dir)


def extract_sailvos3d(target_dir, extract_dir, n_workers):
    """
    Extract the SAIL-VOS 3D dataset following its specific extraction process.
    This function runs all extraction stages in sequence.

    Args:
        target_dir (str): Directory containing the downloaded files
        extract_dir (str): Directory to extract files to
        n_workers (int): Number of parallel workers for extraction

    Returns:
        bool: True if extraction was successful
    """
    print(f"Extracting SAIL-VOS 3D dataset to {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)

    # Run all extraction stages in sequence
    if not untar_sailvos3d(target_dir, n_workers):
        return False

    if not extract_zips_sailvos3d(target_dir, extract_dir, n_workers):
        return False

    if not organize_files_sailvos3d(target_dir, extract_dir):
        return False

    if not extract_mesh_zips_sailvos3d(target_dir):
        return False

    if not move_meshes_sailvos3d(target_dir, extract_dir):
        return False

    print("Complete extraction finished!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and process SAIL-VOS 3D dataset"
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
        help="Number of parallel workers for downloading",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "download",
            "extract",
            "untar",
            "extract_zips",
            "organize_files",
            "extract_mesh_zips",
            "move_meshes",
            "all",
        ],
        default=["all"],
        help="Stages to perform: 'download', 'extract' (all extraction stages), 'untar', 'extract_zips', 'extract_mesh_zips', 'move_meshes', 'organize_files', or 'all'",
    )

    args = parser.parse_args()

    # Set default directories if not provided
    if args.extract_dir is None:
        args.extract_dir = os.path.join(args.target_dir, "extracted")

    # Determine which operations to perform
    do_download = "download" in args.stages or "all" in args.stages
    do_extract = "extract" in args.stages or "all" in args.stages
    do_untar = "untar" in args.stages or "all" in args.stages
    do_extract_zips = "extract_zips" in args.stages or "all" in args.stages
    do_organize_files = "organize_files" in args.stages or "all" in args.stages
    do_extract_mesh_zips = "extract_mesh_zips" in args.stages or "all" in args.stages
    do_move_meshes = "move_meshes" in args.stages or "all" in args.stages

    # Execute the requested operations
    if do_download:
        if not download_sailvos3d(args.target_dir, args.n_workers):
            print("Download failed. Exiting.")
            return 1

    if do_extract:
        # Run all extraction stages
        if not extract_sailvos3d(args.target_dir, args.extract_dir, args.n_workers):
            print("Extraction failed. Exiting.")
            return 1
    else:
        # Run individual extraction stages if specified
        if do_untar:
            if not untar_sailvos3d(args.target_dir, args.n_workers):
                print("Untar stage failed. Exiting.")
                return 1

        if do_extract_zips:
            if not extract_zips_sailvos3d(
                args.target_dir, args.extract_dir, args.n_workers
            ):
                print("Zip extraction stage failed. Exiting.")
                return 1

        if do_organize_files:
            if not organize_files_sailvos3d(args.target_dir, args.extract_dir):
                print("File organization stage failed. Exiting.")
                return 1

        if do_extract_mesh_zips:
            if not extract_mesh_zips_sailvos3d(args.target_dir):
                print("Mesh zip extraction stage failed. Exiting.")
                return 1

        if do_move_meshes:
            if not move_meshes_sailvos3d(args.target_dir, args.extract_dir):
                print("Mesh file moving stage failed. Exiting.")
                return 1

    print("All operations completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
