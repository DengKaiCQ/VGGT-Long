# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/eval_monodepth.py

import os
import cv2
import sys
import csv
import torch
import argparse
import numpy as np


from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader




sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amb3r.model import AMB3R

from tools import metric
from tools.mono import get_dataset
from tools.alignment import  align_depth_least_square


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/marigold_data/")
    parser.add_argument('--results_path', type=str, default="./outputs/monodepth")
    return parser


def resize_rgb_int_cv2(rgb_int, max_res=518, divisible_by=14, target_resolution=None):
    """
    Resizes a batch of RGB images using OpenCV's bicubic interpolation
    while maintaining the aspect ratio and ensuring dimensions are the CLOSEST
    possible match that is divisible by a given number.

    Args:
        rgb_int (torch.Tensor): A batch of images of shape [B, 3, H, W].
        max_res (int): The maximum resolution for the larger dimension (H or W).
        divisible_by (int): The number each dimension must be divisible by.

    Returns:
        torch.Tensor: The resized batch of images.
    """
    B, C, H, W = rgb_int.shape

    # --- NEW: Check for a direct target_resolution ---
    if target_resolution is not None:
        # If a tuple is provided, use it directly
        new_H, new_W = target_resolution
    else:
        # --- ELSE: Fall back to the previous automatic calculation ---
        long_dim = max(H, W)
        short_dim = min(H, W)

        if long_dim > max_res:
            new_long_dim = (max_res // divisible_by) * divisible_by
        else:
            new_long_dim = (long_dim // divisible_by) * divisible_by
        
        aspect_ratio = short_dim / long_dim
        new_short_dim_ideal = new_long_dim * aspect_ratio

        short_dim_floor = (int(new_short_dim_ideal) // divisible_by) * divisible_by
        short_dim_ceil = short_dim_floor + divisible_by

        if abs(new_short_dim_ideal - short_dim_floor) < abs(new_short_dim_ideal - short_dim_ceil):
            new_short_dim = short_dim_floor
        else:
            new_short_dim = short_dim_ceil

        if H > W:
            new_H = new_long_dim
            new_W = new_short_dim
        else:
            new_H = new_short_dim
            new_W = new_long_dim
    
    if new_W == 0 or new_H == 0:
        raise ValueError("Calculated new dimensions are zero.")

    resized_images = []
    for i in range(B):
        image_np = rgb_int[i].permute(1, 2, 0).numpy()
        resized_image_np = cv2.resize(
            image_np, (new_W, new_H), interpolation=cv2.INTER_LINEAR
        )

        resized_image_tensor = torch.from_numpy(resized_image_np).permute(2, 0, 1)
        resized_images.append(resized_image_tensor)

    return torch.stack(resized_images)


args = get_args_parser().parse_args()

ckpt_path = '../checkpoints/amb3r.pt'
model = AMB3R()
model.load_weights(ckpt_path)
model.cuda()


os.makedirs(args.results_path, exist_ok=True)


datasets_config = {
    'diode': {
        'cfg': './tools/mono/configs/data_diode_all.yaml',
        'mode': None,
    },
    'kitti_mono': {
        'cfg': './tools/mono/configs/data_kitti_eigen_test.yaml',
        'mode': None,
    },
    'nyu': {
        'cfg': './tools/mono/configs/data_nyu_test.yaml',
        'mode': None,
    },
    'scannet': {
        'cfg': './tools/mono/configs/data_scannet_val.yaml',
        'mode': None,
    },
    'eth3d': {
        'cfg': './tools/mono/configs/data_eth3d.yaml',
        'mode': None,
        'alignment_max_res': 1024
    },
}

eval_metrics = [
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
]


for dataset_name, config in datasets_config.items():
    dataset_config = config['cfg']
    mode = config['mode']
    alignment_max_res = config.get('alignment_max_res', None)

    print(f"Processing dataset: {dataset_name}")

    cfg_data = OmegaConf.load(dataset_config)

    dataset = get_dataset(
        cfg_data, base_data_dir=args.data_path, mode=mode
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]
    tracker = metric.MetricTracker(*[m.__name__ for m in metric_funcs])


    for batch in tqdm(dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True):
        # rgb_int 
        rgb_float = batch["rgb_int"] / 255.0  # [B, 3, H, W]
        rgb_resized = resize_rgb_int_cv2(rgb_float, max_res=518, divisible_by=14, target_resolution=(392, 518))  # [B, 3, H, W]
        rgb_resized = rgb_resized * 2 - 1  # [B, 3, H, W]

        gt_depth_ts = batch["depth_raw_linear"].squeeze().cuda()
        valid_mask_ts = batch["valid_mask_raw"].squeeze().cuda()

        gt_depth = gt_depth_ts.cpu().numpy()
        valid_mask = valid_mask_ts.cpu().numpy()


        view_mono = {
                    'images': rgb_resized[:, None].cuda(non_blocking=True), # [B, 1, 3, H, W]
                }


        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    res = model.run_amb3r_benchmark(view_mono)

        depth1 = res["depth"].squeeze()

        target_h, target_w = gt_depth_ts.shape

        # Resize prediction to match GT resolution
        depth1_np = depth1.cpu().numpy()
        depth1_resized = cv2.resize(depth1_np, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # 1. Align prediction to GT
        aligned_pred, _, _ = align_depth_least_square(
            gt_arr=gt_depth,
            pred_arr=depth1_resized,
            valid_mask_arr=valid_mask,
            return_scale_shift=True,
            max_resolution=alignment_max_res,
        )

        # 2. Clip the aligned prediction
        clipped_pred = np.clip(aligned_pred, a_min=dataset.min_depth, a_max=dataset.max_depth)
        clipped_pred = np.clip(clipped_pred, a_min=1e-6, a_max=None)

        # 3. Calculate all metrics
        pred_ts = torch.from_numpy(clipped_pred).cuda()
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(pred_ts, gt_depth_ts, valid_mask_ts).item()
            tracker.update(_metric_name, _metric)

    results = tracker.result()

    headers = ['Metric', 'depth']
    table_data = [[key, f"{val:.4f}"] for key, val in results.items()]

    print(f"Results for dataset: {dataset_name}")
    # Simple print
    print(f"{headers[0]:<30} | {headers[1]}")
    print("-" * 45)
    for key, val in table_data:
        print(f"{key:<30} | {val}")

    filename = f"{args.results_path}/{dataset_name}_results.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_data)

    print(f"Table data successfully saved to {filename}")