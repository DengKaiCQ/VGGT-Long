#!/bin/bash

if [ -z "$1" ]
then
   echo "Please speficy a target path to this script, e.g.: /setup_monodepth2.sh /path/to/monodepth2";
   exit 1
fi

set -e
TARGET="$1"

echo "Downloading monodepth2 repository https://github.com/nianticlabs/monodepth2 to $TARGET"
mkdir -p "$1"

git clone https://github.com/nianticlabs/monodepth2 $TARGET

OLD_PWD="$PWD"
cd $TARGET

mkdir models
cd models

mkdir mono+stereo_1024x320
cd mono+stereo_1024x320
wget --no-check-certificate https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip
unzip mono+stereo_1024x320.zip
rm mono+stereo_1024x320.zip
cd ..

mkdir mono+stereo_640x192
cd mono+stereo_640x192
wget --no-check-certificate https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip
unzip mono+stereo_640x192.zip
rm mono+stereo_640x192.zip
cd ..

cd "$OLD_PWD"
echo "Done"
exit 0
