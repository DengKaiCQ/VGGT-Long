#!/bin/bash

mkdir weights
cd ./weights

# SALAD (~ 350 MiB)
echo "Downloading SALAD weights..."
SALAD_URL="https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
curl -L "$SALAD_URL" -o "./dino_salad.ckpt"

# DINO (~ 340 MiB)
echo "Downloading DINO weights..."
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# DBoW (~ 40 MiB of tar.gz, ~145 MiB of txt)
echo "Downloading DBoW weights..."
(wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz) & wait
tar -xzvf ORBvoc.txt.tar.gz
rm ORBvoc.txt.tar.gz

# VGGT (~ 5.0 GiB)
echo "Downloading VGGT weights..."
VGGT_URL="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
curl -L "$VGGT_URL" -o "./model.pt"

# you will see 4 files under `./weights` when finished
# - model.pt
# - dino_salad.ckpt             
# - dinov2_vitb14_pretrain.pth  
# - ORBvoc.txt
