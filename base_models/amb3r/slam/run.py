# References:
#   https://github.com/HengyiWang/amb3r/blob/main/slam/run.py

import sys
import os
import time
import torch
import argparse
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from amb3r.model import AMB3R
from amb3r.tools.pts_vis import get_pts_mask

from slam.pipeline import AMB3R_VO
from slam.datasets.tum_slam import Tum
from slam.datasets.demo import DemoDataset



torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 for matmul on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./demo")
    parser.add_argument('--demo_name', type=str, default="demo")
    parser.add_argument('--results_path', type=str, default="./outputs/slam/demo")
    parser.add_argument('--target_point_count', type=int, default=3_000_000)
    parser.add_argument('--edge_mask', type=bool, default=True)
    parser.add_argument('--conf_threshold', type=float, default=0.0)
    parser.add_argument('--save_res', type=bool, default=True)

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    data = DemoDataset(ROOT=args.data_path, resolution=(518, 392))


    os.makedirs(args.results_path, exist_ok=True)
    model = AMB3R()
    model.load_weights('./checkpoints/amb3r.pt')
            
    model.cuda()
    model.eval()
    pipeline = AMB3R_VO(model)


    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    _, views_all = batch
    num_frames = views_all['images'].shape[1]


    print(f"Processing scene {args.demo_name} with {num_frames} frames...")

    images = views_all['images'] # (1, T, 3, H, W), in [-1, 1] range

    start_time = time.time()
    res  = pipeline.run(images)
    end_time = time.time()
    fps = num_frames / (end_time - start_time)
    print(f"Processed {num_frames} frames with {fps} fps")

    poses_pred = res.poses
    pts_pred = res.pts
    conf_pred = res.conf
    kf_idx = res.kf_idx
    
    if args.edge_mask:
        pts_mask, sky_mask = get_pts_mask(pts_pred, views_all['images'], conf_pred, conf_threshold=args.conf_threshold)
    else:
        pts_mask = conf_pred >= args.conf_threshold
        sky_mask = torch.zeros_like(pts_mask)

    # Save pts
    pts_kf = pts_pred[kf_idx].cpu().numpy().reshape(-1, 3)
    color_kf = (images[0, kf_idx].cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 3) + 1.0) / 2.0
    
    pts_mask_kf = pts_mask[kf_idx].reshape(-1)
    pts_kf = pts_kf[pts_mask_kf]
    color_kf = color_kf[pts_mask_kf]

    if len(pts_kf) > args.target_point_count:
        print(f"Downsampling keyframe points from {len(pts_kf)} to {args.target_point_count} points...")
        sampled_indices = np.random.choice(len(pts_kf), args.target_point_count, replace=False)
        pts_kf = pts_kf[sampled_indices]
        color_kf = color_kf[sampled_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_kf)
    pcd.colors = o3d.utility.Vector3dVector(color_kf)
    o3d.io.write_point_cloud(os.path.join(args.results_path, f"scene_{args.demo_name}_keyframe_points.ply"), pcd)
    
    if args.save_res:
        res_save = {
                'pts': pts_pred.cpu().numpy(),
                'conf': conf_pred.cpu().numpy(),
                'pose': poses_pred.cpu().numpy(),
                'images': views_all['images'].cpu().squeeze(0).numpy(),
                'sky_mask': sky_mask,
                'kf_idx': kf_idx.cpu().numpy(),
                'fps': fps,
            }
        save_path = os.path.join(args.results_path, f"scene_{args.demo_name}_results.npz")
        np.savez_compressed(save_path, **res_save)
        print(f"Saved results to {save_path}")
    

if __name__ == "__main__":
    main()
