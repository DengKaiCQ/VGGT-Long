#!/bin/bash

SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

if [ -z "$1" ]
then
   echo "Please speficy a target path to this script, e.g.: /setup_cvp_mvsnet.sh /path/to/cvp_mvsnet";
   exit 1
fi

set -e
TARGET="$1"

echo "Downloading CVP-MVSNet repository https://github.com/JiayuYANG/CVP-MVSNet to $TARGET"
mkdir -p "$1"

git clone https://github.com/JiayuYANG/CVP-MVSNet $TARGET

OLD_PWD="$PWD"
cd $TARGET

# apply patch:
cd CVP_MVSNet
cd models

PATCH_FILE=${SCRIPT_DIR}/cvp_mvsnet.patch
echo "Applying patch ${PATCH_FILE} to modules.py"
patch modules.py < $PATCH_FILE

cd "$OLD_PWD"
echo "Done"
exit 0
