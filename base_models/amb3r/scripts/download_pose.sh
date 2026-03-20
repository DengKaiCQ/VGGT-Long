#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
POSE_DIR="${PROJECT_ROOT}/data/pose"

# Create destination directory
echo "Creating destination directory at $POSE_DIR..."
mkdir -p "$POSE_DIR"

# Download the zip file
FILE_URL="https://huggingface.co/datasets/HengyiWang/re10k_amb3r_split/resolve/main/re10k_amb3r_split.zip"
FILE_NAME="re10k_amb3r_split.zip"

echo "Downloading ${FILE_NAME}..."
wget -c "$FILE_URL" -P "$POSE_DIR"

# Unzip the file
echo "Unzipping ${FILE_NAME}..."
unzip -q "${POSE_DIR}/${FILE_NAME}" -d "$POSE_DIR"

# Clean up
echo "Cleaning up ${FILE_NAME}..."
rm "${POSE_DIR}/${FILE_NAME}"

echo "Done! All pose data saved in $POSE_DIR."