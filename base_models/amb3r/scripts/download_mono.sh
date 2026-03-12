#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Navigate from the scripts folder to the parent directory
cd ..

# 2. Enter the existing data folder and create the marigold_data directory
echo "Navigating to data folder and creating marigold_data directory..."
mkdir -p data/marigold_data
cd data/marigold_data

# 3. Download the dataset using wget
# -r: recursive download
# -np: don't ascend to the parent directory
# -nH: disable generation of host-prefixed directories
# --cut-dirs=4: skip the 4 URL directories (~pf/bingkedata/marigold/evaluation_dataset/)
# -R "index.html*": reject auto-generated index.html files
echo "========================================"
echo "Starting download for Marigold dataset..."
echo "Target directory: $(pwd)"

wget -r -np -nH --cut-dirs=4 -R "index.html*" https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/

echo "========================================"
echo "Marigold dataset download completed successfully!"