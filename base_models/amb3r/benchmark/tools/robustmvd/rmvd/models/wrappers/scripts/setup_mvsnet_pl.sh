#!/bin/bash

if [ -z "$1" ]
then
   echo "Please speficy a target path to this script, e.g.: /setup_mvsnet_pl.sh /path/to/mvsnet_pl";
   exit 1
fi

set -e
TARGET="$1"

echo "Downloading MVSNet_pl repository https://github.com/kwea123/MVSNet_pl to $TARGET"
mkdir -p "$1"

git clone https://github.com/kwea123/MVSNet_pl $TARGET

OLD_PWD="$PWD"
cd $TARGET

wget --no-check-certificate https://github.com/kwea123/MVSNet_pl/releases/download/v1.0/_ckpt_epoch_14.ckpt

# apply patch:
cd models
sed -i -e 's/prob_volume = F.softmax(cost_reg, 1) # (B, D, h, w)/prob_volume = torch.nan_to_num(F.softmax(cost_reg, 1)) # (B, D, h, w)/g' mvsnet.py

cd "$OLD_PWD"
echo "Done"
exit 0
