# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/eval_videodepth.py

import os
import cv2
import sys
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amb3r.model import AMB3R
from amb3r.datasets import Bonn, Sintel, Kitti
from tools.depth_eval import depth_evaluation

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/dynamic/")
    parser.add_argument('--results_path', type=str, default="./outputs/videodepth")
    return parser

args = get_args_parser().parse_args()

ckpt_path = '../checkpoints/amb3r.pt'
model = AMB3R()
model.load_weights(ckpt_path)
model.cuda()

os.makedirs(args.results_path, exist_ok=True)

eval_datasets_all = {
    'sintel': Sintel(ROOT=args.data_path + '/sintel',
                     resolution=(518, 392), full_video=True, kf_every=1),
    'bonn': Bonn(ROOT=args.data_path + '/bonn',
                 resolution=(518, 392), full_video=True, kf_every=1),
    'kitti': Kitti(ROOT=args.data_path + '/kitti',
                            resolution=(518, 392), full_video=True, kf_every=1),
}

with torch.no_grad():
    for dataset_name, data in eval_datasets_all.items():
        dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)

        total_depth_result = {}
        batch_count = 0

        for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
            views, views_all = batch

            for key in views_all.keys():
                if key == 'images':
                    views_all[key] = views_all[key].cuda(non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                res = model.run_amb3r_benchmark(views_all)

            gt_depths = views_all['depthmap_ori'].permute(1, 0, 2, 3).squeeze(1)  # T, H_ori, W_ori
            depth1 = res["depth"].squeeze().cpu()  # T, H, W

            del res

            # Resize depth1 to match GT resolution
            target_h, target_w = gt_depths.shape[1:]
            depth_np = depth1.numpy()
            resized_frames = []
            for f in range(depth_np.shape[0]):
                resized = cv2.resize(depth_np[f], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                resized_frames.append(resized)
            depth1 = torch.from_numpy(np.stack(resized_frames, axis=0))

            # Dataset-specific eval args
            if dataset_name == 'bonn':
                eval_args = {'max_depth': 70, 'use_gpu': True}
            elif dataset_name == 'sintel':
                eval_args = {'max_depth': 70, 'use_gpu': True, 'post_clip_max': 70}
            elif dataset_name == 'kitti_mono':
                eval_args = {'max_depth': None, 'use_gpu': True}
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                depth1.cpu(), gt_depths.cpu(), **eval_args
            )

            # Accumulate results
            if not total_depth_result:
                total_depth_result = {key: 0.0 for key in depth_results}
            for key in depth_results:
                total_depth_result[key] += depth_results[key]
            batch_count += 1

            print(f"Batch {batch_count} - Abs Rel: {total_depth_result['Abs Rel'] / batch_count:.4f}")

            # Cleanup
            for key in views_all.keys():
                if torch.is_tensor(views_all[key]) and views_all[key].is_cuda:
                    views_all[key] = views_all[key].cpu()

            del views, views_all, gt_depths, depth1, depth_predict, error_map, depth_gt

        # Print final results
        avg_depth_result = {key: value / batch_count for key, value in total_depth_result.items()}
        print(f"\nFinal Depth Results ({dataset_name}):")
        for key, value in avg_depth_result.items():
            print(f"  {key}: {value:.4f}")

        # Save results
        with open(os.path.join(args.results_path, f"{dataset_name}_videodepth.json"), 'w') as f:
            json.dump({
                "depth_results": avg_depth_result,
                "batch_count": batch_count,
            }, f, indent=4)
