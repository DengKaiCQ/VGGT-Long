#!/bin/bash

# Define paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DYNAMIC_DIR="${PROJECT_ROOT}/data/dynamic"

# ============================================================
# Sintel
# ============================================================
SINTEL_DIR="${DYNAMIC_DIR}/sintel"
mkdir -p "$SINTEL_DIR"
echo "Downloading Sintel..."

wget -c --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip -P "$SINTEL_DIR"
wget -c --no-proxy http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip -P "$SINTEL_DIR"
wget -c --no-proxy http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_extras.zip -P "$SINTEL_DIR"

echo "Unzipping Sintel..."
find "$SINTEL_DIR" -name "*.zip" -exec unzip -o -q {} -d "$SINTEL_DIR" \;
find "$SINTEL_DIR" -name "*.zip" -exec rm {} \;

echo "Sintel done!"

# ============================================================
# Bonn
# ============================================================
BONN_DIR="${DYNAMIC_DIR}/bonn"
mkdir -p "$BONN_DIR"
echo "Downloading Bonn..."

wget -c https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_dataset.zip -P "$BONN_DIR"

echo "Unzipping Bonn..."
unzip -o -q "${BONN_DIR}/rgbd_bonn_dataset.zip" -d "$BONN_DIR"
rm "${BONN_DIR}/rgbd_bonn_dataset.zip"

echo "Preprocessing Bonn..."
python "${PROJECT_ROOT}/benchmark/data/bonn_preprocessing.py" --data_path "$BONN_DIR"

echo "Bonn done!"

# ============================================================
# KITTI
# ============================================================
KITTI_DIR="${DYNAMIC_DIR}/kitti"
mkdir -p "$KITTI_DIR"
echo "Downloading KITTI..."

wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0013/2011_09_26_drive_0013_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0020/2011_09_26_drive_0020_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0023/2011_09_26_drive_0023_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0036/2011_09_26_drive_0036_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0079/2011_09_26_drive_0079_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0095/2011_09_26_drive_0095_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0113/2011_09_26_drive_0113_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0037/2011_09_28_drive_0037_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0026/2011_09_29_drive_0026_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip -P "$KITTI_DIR"
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip -P "$KITTI_DIR"

echo "Unzipping KITTI..."
find "$KITTI_DIR" -name "*.zip" -exec unzip -o -q {} -d "$KITTI_DIR" \;
find "$KITTI_DIR" -name "*.zip" -exec rm {} \;

echo "Preprocessing KITTI..."
python "${PROJECT_ROOT}/benchmark/data/kitti_preprocessing.py" --data_path "$KITTI_DIR"

echo "KITTI done!"

echo "Done! All dynamic datasets are saved in $DYNAMIC_DIR."