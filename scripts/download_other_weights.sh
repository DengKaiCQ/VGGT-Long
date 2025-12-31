#!/bin/bash

cd ./weights

mkdir Pi3
cd ./Pi3

# Pi3 (~ 3.8 GiB)
echo "Downloading Pi3 weights..."
Pi3_URL="https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors"
curl -L "$Pi3_URL" -o "./model.safetensors"

cd ..
mkdir MA
cd ./MA

# MapAnything (~ 2.3 GiB)
echo "Downloading MapAnything weights..."
MA_URL="https://huggingface.co/facebook/map-anything-v1/resolve/main/model.safetensors"
MA_CONFIG_URL="https://huggingface.co/facebook/map-anything-v1/resolve/main/config.json"
curl -L "$MA_URL" -o "./model.safetensors"
curl -L "$MA_CONFIG_URL" -o "./config.json"

cd ..
mkdir DA3
cd ./DA3

# DepthAnything3 (~ 6.8 GiB)
echo "Downloading DepthAnything3 weights..."
DA3_URL="https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main/model.safetensors"
DA3_CONFIG_URL="https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main/config.json"
curl -L "$DA3_URL" -o "./model.safetensors"
curl -L "$DA3_CONFIG_URL" -o "./config.json"

cd ..

# you will see 3 folders under `./weights` when finished
# - Pi3
#    - model.safetensors            
# - MA
#    - model.safetensors  
#    - config.json          
# - DA3
#    - model.safetensors  
#    - config.json  
