#!/bin/bash

if [ -z "$1" ]
then
   echo "Please path a target path to this script, e.g.: /download_staticthings3d.sh /path/to/staticthings3d.";
   exit 1
fi

TARGET="$1"

echo "Downloading StaticThings3D dataset to $TARGET."
mkdir -p "$1"

OLD_PWD="$PWD"
cd $TARGET

wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/StaticThings3D_3DV22/depths.tar.bz2
tar xf depths.tar.bz2
rm depths.tar.bz2

wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/StaticThings3D_3DV22/frames_cleanpass.tar.bz2
tar xf frames_cleanpass.tar.bz2
rm frames_cleanpass.tar.bz2

wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/StaticThings3D_3DV22/frames_finalpass.tar.bz2
tar xf frames_finalpass.tar.bz2
rm frames_finalpass.tar.bz2

wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/StaticThings3D_3DV22/intrinsics.tar.bz2
tar xf intrinsics.tar.bz2
rm intrinsics.tar.bz2

wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/StaticThings3D_3DV22/poses.tar.bz2
tar xf poses.tar.bz2
rm poses.tar.bz2

cd "$OLD_PWD"
echo "Done"
exit 0
