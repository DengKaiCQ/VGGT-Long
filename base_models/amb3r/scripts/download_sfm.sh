#!/bin/bash

# Define paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SFM_DIR="${PROJECT_ROOT}/data/sfm"



# TNT
TNT_DIR="${SFM_DIR}/tnt"
# Create destination directory
mkdir -p "$TNT_DIR"

echo "Downloading Tanks and Temples (ibr3d_tat.tar.gz)..."
wget -c https://storage.googleapis.com/isl-datasets/FreeViewSynthesis/ibr3d_tat.tar.gz -P "$TNT_DIR"

echo "Extracting ibr3d_tat.tar.gz..."
tar -xzf "${TNT_DIR}/ibr3d_tat.tar.gz" -C "$TNT_DIR"

echo "Cleaning up ibr3d_tat.tar.gz..."
rm "${TNT_DIR}/ibr3d_tat.tar.gz"




# IMC 2020
IMC_DIR="${SFM_DIR}/imc"
mkdir -p "$IMC_DIR"

echo "Downloading IMC 2020..."
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/british_museum.tar.gz -P "$IMC_DIR"
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/florence_cathedral_side.tar.gz -P "$IMC_DIR"
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/lincoln_memorial_statue.tar.gz -P "$IMC_DIR"
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/milan_cathedral.tar.gz -P "$IMC_DIR"
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/mount_rushmore.tar.gz -P "$IMC_DIR"
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/piazza_san_marco.tar.gz -P "$IMC_DIR"
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/sagrada_familia.tar.gz -P "$IMC_DIR"
wget -c https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/st_pauls_cathedral.tar.gz -P "$IMC_DIR"

echo "Extracting IMC 2020..."
for archive in "${IMC_DIR}"/*.tar.gz; do
    echo "  Extracting $(basename "$archive")..."
    tar -xzf "$archive" -C "$IMC_DIR"
    echo "  Cleaning up $(basename "$archive")..."
    rm "$archive"
done

echo "Preprocessing IMC 2020..."
python "${PROJECT_ROOT}/benchmark/data/imc_preprocessing.py" --data_path "$IMC_DIR"

echo "Done! SfM datasets are saved in $SFM_DIR."