#!/bin/bash

set -e

usage()
{
   echo "Usage: $0 -o [OPTIONAL] out_base -w [OPTIONAL] -g [OPTIONAL] gpu_list"
   echo -e "\t-o Path to output base directory. Optional. Default: /tmp/rmvd_train"
   echo -e "\t-w Enable wandb logging. Default: Do not enable wandb logging"
   echo -e "\t-g List of space-separated gpu numbers to launch train on (e.g. 0 2 4 5). Optional. Default: 0"
   echo -e "\tNote: the order of the arguments is important"
   exit 1 # Exit program after printing help
}

while getopts "o:wg" opt; do
    case "${opt}" in
        o )
          OUT_BASE=${OPTARG}
          ;;
        w )
          enable_wandb="--wandb"
          ;;
        g )
          gpu_list=${OPTARG}
          ;;
        ? ) usage ;;
    esac
done

if [ -z ${OUT_BASE} ]; then
    OUT_BASE=/tmp/rmvd_train
fi
echo Output base directory: ${OUT_BASE}

shift $((OPTIND-1))
GPU_IDX=("$@")
if [ -z ${GPU_IDX} ]; then
    GPU_IDX=(0)
fi
GPU_IDX_STR=$(printf ",%s" "${GPU_IDX[@]}")
GPU_IDX_STR=${GPU_IDX_STR:1}
NUM_GPUS=${#GPU_IDX[@]}
echo Using ${NUM_GPUS} GPUs with indices: ${GPU_IDX[@]}
export CUDA_VISIBLE_DEVICES=${GPU_IDX_STR}

echo

# robust_mvd model:
python train.py --training_type mvd --output ${OUT_BASE}/robust_mvd --num_gpus ${NUM_GPUS} --batch_size 4 --max_iterations 600000 --model robust_mvd --inputs poses intrinsics --optimizer adam --lr 1e-4 --grad_clip_max_norm 5 --scheduler flownet_scheduler --loss robust_mvd_loss --dataset staticthings3d.robust_mvd.mvd --dataset blendedmvs.robust_mvd.mvd --augmentations_per_dataset robust_mvd_augmentations_staticthings3d --augmentations_per_dataset robust_mvd_augmentations_blendedmvs --batch_augmentations robust_mvd_batch_augmentations --seed 42 ${enable_wandb}
