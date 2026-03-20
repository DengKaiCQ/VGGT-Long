#!/bin/bash

if [ -z "$1" ]
then
   echo "Please path a target path to this script, e.g.: /download_dtu.sh /path/to/dtu.";
   exit 1
fi

TARGET="$1"

echo "Downloading DTU dataset to $TARGET."
mkdir -p "$1"

OLD_PWD="$PWD"
cd $TARGET

# 1. Download the file `dtu.tar.xz` from https://polybox.ethz.ch/index.php/s/ugDdJQIuZTk4S35 (supplied by the
# PatchmatchNet repository https://github.com/FangjinhuaWang/PatchmatchNet) and extract it.
wget --no-check-certificate "https://polybox.ethz.ch/index.php/s/ugDdJQIuZTk4S35/download" -O dtu.tar.xz
tar xf dtu.tar.xz
rm dtu.tar.xz

# 2. Download the rectified images from the original dataset website: 
# http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip and extract it.
wget --no-check-certificate "http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip"
unzip Rectified.zip
rm Rectified.zip

# 3. Download the pointclouds from the original dataset website: 
# http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip and extract it.
wget --no-check-certificate "http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"
unzip Points.zip
rm Points.zip

cd "$OLD_PWD"
echo "Done"
exit 0
