#!/bin/bash

# Define paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RMVD_DIR="${PROJECT_ROOT}/data/rmvd/7scenes"

# Create destination directory
mkdir -p "$RMVD_DIR"

# List of scenes
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

# Base URL
base_url="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"

# Loop through each scene
for scene in "${scenes[@]}"; do
    echo "Downloading ${scene}..."
    wget -c "${base_url}/${scene}.zip" -P "$RMVD_DIR"
    
    echo "Unzipping ${scene}..."
    unzip -q "${RMVD_DIR}/${scene}.zip" -d "$RMVD_DIR"
    
    echo "Cleaning up ${scene}.zip..."
    rm "${RMVD_DIR}/${scene}.zip"
    
    echo "Unzipping sequence files for ${scene}..."
    for seq_zip in "${RMVD_DIR}/${scene}"/*.zip; do
        # Check if the file exists to handle cases where there are no zip files
        if [ -f "$seq_zip" ]; then
            echo "  Extracting $(basename "$seq_zip")..."
            unzip -q "$seq_zip" -d "${RMVD_DIR}/${scene}"
            
            echo "  Cleaning up $(basename "$seq_zip")..."
            rm "$seq_zip"
        fi
    done
done

echo "Extracting SfM poses used in AMB3R..."
POSES_ZIP="${PROJECT_ROOT}/benchmark/data/7scenes_sfm_poses.zip"
if [ -f "$POSES_ZIP" ]; then
    unzip -q "$POSES_ZIP" -d "$RMVD_DIR"
else
    echo "Warning: $POSES_ZIP not found. Skipping sfm_poses extraction."
fi

echo "Preprocessing 7scenes depth maps..."
python "${PROJECT_ROOT}/benchmark/data/7scenes_preprocessing.py" --data_path "$RMVD_DIR"

echo "Done! All datasets are saved in $RMVD_DIR."