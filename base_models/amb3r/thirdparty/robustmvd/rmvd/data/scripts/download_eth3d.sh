#!/bin/bash

if [ -z "$1" ]
then
   echo "Please path a target path to this script, e.g.: /download_eth3d.sh /path/to/eth3d.";
   exit 1
fi

TARGET="$1"

echo "Downloading ETH3D dataset to $TARGET."
mkdir -p "$1"

OLD_PWD="$PWD"
cd $TARGET

categories=(courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains)
datas=(dslr_jpg dslr_undistorted dslr_depth)  # dslr_raw scan_raw scan_clean scan_eval dslr_occlusion

for category in ${categories[@]}; do
  for data in ${datas[@]}; do
    filename=${category}_${data}.7z
    wget --no-check-certificate "https://www.eth3d.net/data/${filename}"
    7z x ${filename}
    rm ${filename}
  done
done

cd "$OLD_PWD"
echo "Done"
exit 0
