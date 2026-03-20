#!/bin/bash

set -e

usage()
{
   echo "Usage: $0 -o [OPTIONAL] out_base -n [OPTIONAL] num_samples -u [OPTIONAL] -g [OPTIONAL] gpu_list"
   echo -e "\t-o Path to output base directory. Optional. Default: /tmp/rmvd_eval"
   echo -e "\t-n Number of samples to evaluate per dataset. Optional. Default: Evaluate all samples"
   echo -e "\t-u Evaluate Uncertainty. Default: Do not evaluate uncertainty"
   echo -e "\t-g List of space-separated gpu numbers to launch train on (e.g. 0 2 4 5). Optional. Default: 0"
   echo -e "\tNote: the order of the arguments is important"
   exit 1 # Exit program after printing help
}

while getopts "o:n:ug" opt; do
    case "${opt}" in
        o )
          OUT_BASE=${OPTARG}
          ;;
        n )
          num_samples="--num_samples ${OPTARG}"
          ;;
        u )
          eval_uncertainty="--eval_uncertainty"
          ;;
        g )
          gpu_list=${OPTARG}
          ;;
        ? ) usage ;;
    esac
done

if [ -z ${OUT_BASE} ]; then
    OUT_BASE=/tmp/rmvd_eval
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
python eval.py --eval_type robustmvd --model robust_mvd --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}

# robust_mvd_5M model:
python eval.py --eval_type robustmvd --model robust_mvd_5M --inputs poses intrinsics --output ${OUT_BASE}/robust_mvd_5M --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}

# monodepth2_mono_stereo_1024x320_wrapped model:
python eval.py --eval_type robustmvd --model monodepth2_mono_stereo_1024x320_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_1024x320_wrapped --num_gpus ${NUM_GPUS} --max_source_views 0 --alignment median ${num_samples}

# monodepth2_mono_stereo_640x192_wrapped model:
python eval.py --eval_type robustmvd --model monodepth2_mono_stereo_640x192_wrapped --output ${OUT_BASE}/monodepth2_mono_stereo_640x192_wrapped --num_gpus ${NUM_GPUS} --max_source_views 0 --alignment median ${num_samples}

# mvsnet_pl_wrapped model:
python eval.py --eval_type robustmvd --model mvsnet_pl_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/mvsnet_pl_wrapped/known_depth_range --exp_name known_depth_range --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model mvsnet_pl_wrapped --inputs poses intrinsics --output ${OUT_BASE}/mvsnet_pl_wrapped/unknown_depth_range --exp_name unknown_depth_range --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}

# midas_big_v2_1_wrapped model:
python eval.py --eval_type robustmvd --model midas_big_v2_1_wrapped --output ${OUT_BASE}/midas_big_v2_1_wrapped --num_gpus ${NUM_GPUS} --max_source_views 0 --alignment least_squares_scale_shift ${num_samples}

# vis_mvsnet_wrapped model
python eval.py --eval_type robustmvd --model vis_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/vis_mvsnet_wrapped/known_depth_range --exp_name known_depth_range --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model vis_mvsnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/vis_mvsnet_wrapped/unknown_depth_range --exp_name unknown_depth_range --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 896 1216 --scannet_size 448 640 --tanks_and_temples_size 704 1280 ${num_samples} ${eval_uncertainty}

# cvp_mvsnet_wrapped model
python eval.py --eval_type robustmvd --model cvp_mvsnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/cvp_mvsnet_wrapped/known_depth_range --exp_name known_depth_range --min_source_views 2 --view_ordering nearest --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 1184 1600 --scannet_size 448 640 --tanks_and_temples_size 1088 1952 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model cvp_mvsnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/cvp_mvsnet_wrapped/unknown_depth_range --exp_name unknown_depth_range --min_source_views 2 --view_ordering nearest --num_gpus ${NUM_GPUS} --eth3d_size 768 1152 --kitti_size 384 1280 --dtu_size 1184 1600 --scannet_size 448 640 --tanks_and_temples_size 1088 1952 ${num_samples} ${eval_uncertainty}

# patchmatchnet_wrapped model
python eval.py --eval_type robustmvd --model patchmatchnet_wrapped --inputs poses intrinsics depth_range --output ${OUT_BASE}/patchmatchnet_wrapped/known_depth_range --exp_name known_depth_range --num_gpus ${NUM_GPUS} --eth3d_size 1792 2688 --kitti_size 376 1280 --dtu_size 1200 1600 --scannet_size 480 640 --tanks_and_temples_size 1140 2048 ${num_samples} ${eval_uncertainty}
python eval.py --eval_type robustmvd --model patchmatchnet_wrapped --inputs poses intrinsics --output ${OUT_BASE}/patchmatchnet_wrapped/unknown_depth_range --exp_name unknown_depth_range --num_gpus ${NUM_GPUS} --eth3d_size 1792 2688 --kitti_size 376 1280 --dtu_size 1200 1600 --scannet_size 480 640 --tanks_and_temples_size 1140 2048 ${num_samples} ${eval_uncertainty}
