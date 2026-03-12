#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TUM_DIR="${PROJECT_ROOT}/data/slam/tum_scenes"
ETH_DIR="${PROJECT_ROOT}/data/slam/eth_slam"

# echo "Creating TUM directory at $TUM_DIR..."
# mkdir -p "$TUM_DIR"

# # 3. Define all fr1 scenes listed in the image
# SCENES=("360" "desk" "desk2" "floor" "plant" "room" "rpy" "teddy" "xyz")

# # Base URL structure
# BASE_URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg1"

# # 4. Sequentially download, extract, and remove the tgz files
# for SCENE in "${SCENES[@]}"; do
#     FILE_NAME="rgbd_dataset_freiburg1_${SCENE}.tgz"
#     DOWNLOAD_URL="${BASE_URL}/${FILE_NAME}"
    
#     echo "========================================"
#     echo "Processing scene: $SCENE"
#     echo "Download URL: $DOWNLOAD_URL"
    
#     # Download the file (using wget)
#     wget -c "$DOWNLOAD_URL" -P "$TUM_DIR"
    
#     # Extract the file
#     echo "Extracting $FILE_NAME ..."
#     tar -xzf "${TUM_DIR}/${FILE_NAME}" -C "$TUM_DIR"
    
#     # Delete the tgz archive
#     echo "Cleaning up archive $FILE_NAME ..."
#     rm "${TUM_DIR}/${FILE_NAME}"
    
#     echo "Scene $SCENE completed!"
#     echo "========================================"
# done

# echo "All TUM fr1 dataset scenes have been successfully downloaded and extracted!"



# ETH3D

echo "Creating ETH3D directory at $ETH_DIR..."
mkdir -p "$ETH_DIR"

BASE_URL="https://www.eth3d.net/data/slam/datasets"

# List of datasets based on the figure (formatted to match URL paths)
DATASETS=(
    "cables_1"
    "camera_shake_1"
    "einstein_1"
    "plant_1"
    "plant_2"
    "sofa_1"
    "table_3"
    "table_7"
)

FILE_TYPE="mono" # Change to stereo, rgbd, imu, or raw if necessary

for dataset in "${DATASETS[@]}"; do
    FILE="${dataset}_${FILE_TYPE}.zip"
    DOWNLOAD_URL="${BASE_URL}/${FILE}"
    
    echo "========================================"
    echo "Processing scene: $dataset"
    echo "Download URL: $DOWNLOAD_URL"
    
    # Download the file (using wget)
    wget -c "$DOWNLOAD_URL" -P "$ETH_DIR"
    
    # Extract the file
    echo "Extracting $FILE ..."
    unzip -q "${ETH_DIR}/${FILE}" -d "$ETH_DIR"
    
    # Delete the zip archive
    echo "Cleaning up archive $FILE ..."
    rm "${ETH_DIR}/${FILE}"
    
    echo "Scene $dataset completed!"
    echo "========================================"
done

echo "All ETH3D dataset scenes have been successfully downloaded and extracted!"