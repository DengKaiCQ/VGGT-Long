# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/eval_pose.py

import os
import sys
import torch
import argparse
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amb3r.model import AMB3R
from amb3r.datasets.rel10k import Re10kAMB3R

from tools.pose_eval import get_results_from_camera_pose

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/pose/re10k_amb3r_split/")
    parser.add_argument('--results_path', type=str, default="./outputs/pose")
    return parser

args = get_args_parser().parse_args()

ckpt_path = '../checkpoints/amb3r.pt'
model = AMB3R()
model.load_weights(ckpt_path)
model.cuda()

os.makedirs(args.results_path, exist_ok=True)




data = Re10kAMB3R(split='test', ROOT=args.data_path, 
                resolution=(518, 294))
dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=2)


total_result1 = {}
batch_count = 0

for i, batch in enumerate(tqdm(dataloader)):
    views, views_all = batch

    for key in views_all.keys():
        views_all[key] = views_all[key].cuda(non_blocking=True) 

    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            res = model.run_amb3r_benchmark(views_all)

    
    with torch.cuda.amp.autocast(dtype=torch.float64):
    
        pose = res["pose"].squeeze(0)
        gt_pose = views_all['camera_pose'].squeeze(0)
        
        result, extri, gt_extri  = get_results_from_camera_pose(pose, gt_pose)

        # Initialize accumulators on the first run
        if not total_result1:
            total_result1 = {key: 0.0 for key in result}

        # Accumulate results
        for key in result:
            total_result1[key] += result[key]

        batch_count += 1

        running_avg1_mAA_30 = total_result1['mAA_30'] / batch_count if batch_count > 0 else 0.0

        print(f"{views[0]['instance']}:Batch {i+1}/{len(dataloader)} -> mAA_30 (res1): {result['mAA_30']:.4f} (running avg: {running_avg1_mAA_30:.4f})")

        # Record image paths for this sample on the fly
        instance = views[0]['instance'][0] if isinstance(views[0]['instance'], (list, tuple)) else views[0]['instance']
        labels = [v['label'][0] if isinstance(v['label'], (list, tuple)) else v['label'] for v in views]
        
    del res, views_all
    torch.cuda.empty_cache()


# --- Averaging and Displaying Results ---
if batch_count > 0:
    avg_result = {key: val / batch_count for key, val in total_result1.items()}

    print("\n-------------------------------------------")
    print(f"Average Stats over {batch_count} batches")
    print("-------------------------------------------")

    print("\n--- Average Result ---")
    for key, val in avg_result.items():
        print(f"{key:<12}: {val:.4f}")
else:
    print("No data was processed. The dataloader might be empty.")


# Save averge results to a text file
with open(os.path.join(args.results_path, f"pose_results.txt"), 'w') as f:
    f.write(f"Average Stats over {batch_count} batches\n")
    f.write("-------------------------------------------\n")

    f.write("\n--- Average Result ---\n")
    for key, val in avg_result.items():
        f.write(f"{key:<12}: {val:.4f}\n")