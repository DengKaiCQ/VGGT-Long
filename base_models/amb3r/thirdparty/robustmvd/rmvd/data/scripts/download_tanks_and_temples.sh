#!/bin/bash

if [ -z "$1" ]
then
   echo "Please path a target path to this script, e.g.: /download_tanks_and_temples.sh /path/to/tanks_and_temples.";
   exit 1
fi

TARGET="$1"

echo "Downloading Tanks and Temples dataset to $TARGET."
mkdir -p "$1"

OLD_PWD="$PWD"
cd $TARGET

wget --no-check-certificate "https://lmb.informatik.uni-freiburg.de/data/robustmvd/tanks_and_temples_images.zip"
unzip tanks_and_temples_images.zip
rm tanks_and_temples_images.zip

wget --no-check-certificate "https://lmb.informatik.uni-freiburg.de/data/robustmvd/tanks_and_temples_depth.zip"
unzip tanks_and_temples_depth.zip
rm tanks_and_temples_depth.zip

cd "$OLD_PWD"
echo "Done"
exit 0
