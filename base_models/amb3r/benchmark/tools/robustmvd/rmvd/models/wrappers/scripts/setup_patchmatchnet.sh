#!/bin/bash

if [ -z "$1" ]
then
   echo "Please speficy a target path to this script, e.g.: /setup_patchmatchnet.sh /path/to/patchmatchnet";
   exit 1
fi

set -e
TARGET="$1"

echo "Downloading PatchmatchNet repository https://github.com/FangjinhuaWang/PatchmatchNet to $TARGET"
mkdir -p "$1"

git clone https://github.com/FangjinhuaWang/PatchmatchNet $TARGET

echo "Done"
exit 0
