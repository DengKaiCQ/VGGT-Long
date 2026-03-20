# References:
#   https://github.com/HengyiWang/amb3r/blob/main/sfm/run.py

import os
import sys
import torch
import argparse
import numpy as np
import open3d as o3d



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'thirdparty'))

from torch.utils.data import DataLoader

from sfm.pipeline import AMB3R_SfM
from amb3r.datasets.demo import Demo
from amb3r.model import AMB3R
from amb3r.tools.pts_vis import get_pts_mask


torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 for matmul on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True




def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./demo")
    parser.add_argument('--demo_name', type=str, default="demo")
    parser.add_argument('--results_path', type=str, default="./outputs/sfm/demo")
    parser.add_argument('--target_point_count', type=int, default=3_000_000)
    parser.add_argument('--edge_mask', type=bool, default=True)
    parser.add_argument('--conf_threshold', type=float, default=0.0)
    parser.add_argument('--save_res', type=bool, default=True)

    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    os.makedirs(args.results_path, exist_ok=True)


    data = Demo(ROOT=args.data_path, resolution=(518, 392), kf_every=1, full_video=True)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)


    model = AMB3R()
    model.load_weights('./checkpoints/amb3r.pt')
            
    model.cuda()
    model.eval()
    pipeline = AMB3R_SfM(model)
    
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    _, views_all = batch
    num_frames = views_all['images'].shape[1]


    scene_idx = args.demo_name

    print(f"Processing scene {scene_idx} with {num_frames} frames.")

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            res = pipeline.run(views_all['images'])
    
    poses_pred = res.poses
    pts_pred = res.pts
    conf = res.conf
    kf_idx = res.kf_idx
    unmapped_frames = res.unmapped_frames  # set of frame indices skipped during coarse reg


    # save the results — exclude unmapped frames from point cloud
    mapped_mask = torch.ones(num_frames, dtype=torch.bool)
    if unmapped_frames:
        print(f"Excluding {len(unmapped_frames)} unmapped frames from point cloud: {sorted(unmapped_frames)}")
        for idx in unmapped_frames:
            mapped_mask[idx] = False

    pts_pred_mapped = pts_pred[mapped_mask]
    images_mapped = views_all['images'][0][mapped_mask]

    if args.edge_mask:
        pts_mask, sky_mask = get_pts_mask(pts_pred, views_all['images'], conf, conf_threshold=args.conf_threshold)
    else:
        pts_mask = conf >= args.conf_threshold
        _, sky_mask = get_pts_mask(pts_pred, views_all['images'], conf, conf_threshold=args.conf_threshold)

    pts_mask_mapped = pts_mask[mapped_mask]

    pts_masked = pts_pred_mapped[pts_mask_mapped]
    images_vis = images_mapped.permute(0, 2, 3, 1)
    color_masked = images_vis[pts_mask_mapped]
    color_masked = (color_masked.cpu().numpy() + 1.0) / 2.0
    
    pts_masked = pts_masked.cpu().numpy()

    if len(pts_masked) > args.target_point_count:
        print(f"Downsampling keyframe points from {len(pts_masked)} to {args.target_point_count} points...")
        sampled_indices = np.random.choice(len(pts_masked), args.target_point_count, replace=False)
        pts_masked = pts_masked[sampled_indices]
        color_masked = color_masked[sampled_indices]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_masked)
    pcd.colors = o3d.utility.Vector3dVector(color_masked)
    o3d.io.write_point_cloud(os.path.join(args.results_path, f"scene_{scene_idx}_points.ply"), pcd)
    
    
    if args.save_res:
        res_save = {
                'pts': pts_pred.cpu().numpy(),
                'conf': conf.cpu().numpy(),
                'pose': poses_pred.cpu().numpy(),
                'images': views_all['images'].cpu().squeeze(0).numpy(),
                'sky_mask': sky_mask,
                'kf_idx': np.array(kf_idx),
                'unmapped_frames': np.array(sorted(unmapped_frames), dtype=np.int64),
            }
        save_path = os.path.join(args.results_path, f"scene_{args.demo_name}_results.npz")
        np.savez_compressed(save_path, **res_save)
        print(f"Saved results to {save_path}")


if __name__ == "__main__":
    main()  
