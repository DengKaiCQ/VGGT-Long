#!/bin/bash

if [ -z "$1" ]
then
   echo "Please speficy a target path to this script, e.g.: /setup_vis_mvsnet.sh /path/to/vis_mvsnet";
   exit 1
fi

set -e
TARGET="$1"

echo "Downloading Vis-MVSNet repository https://github.com/jzhangbs/Vis-MVSNet.git to $TARGET"
mkdir -p "$1"

git clone https://github.com/jzhangbs/Vis-MVSNet.git $TARGET

echo "Done"
exit 0
