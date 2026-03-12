# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/eval_slam.py

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
from amb3r.datasets import SevenScenes
from amb3r.tools.pts_vis import get_pts_mask

from slam.pipeline import AMB3R_VO
from slam.datasets.tum_slam import Tum

from benchmark.tools.pose_eval import evaluate_evo

torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 for matmul on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/")
    parser.add_argument('--results_path', type=str, default="./outputs/slam")
    parser.add_argument('--ckpt_path', type=str, default="../checkpoints/amb3r.pt")
    parser.add_argument('--num_iters', type=int, default=2)
    parser.add_argument('--target_point_count', type=int, default=3_000_000)
    parser.add_argument('--conf_threshold', type=float, default=0.0)
    return parser

args = get_args_parser().parse_args()

model = AMB3R()
model.load_weights(args.ckpt_path)
        
model.cuda()
model.eval()
pipeline = AMB3R_VO(model, cfg_path="../slam/slam_config.yaml")

os.makedirs(args.results_path, exist_ok=True)

eval_datasets_all = {
    '7scenes': SevenScenes(split='test', ROOT=args.data_path + 'rmvd/7scenes/', 
                           resolution=(518, 392), kf_every=1, slam=True),
    'tum': Tum(ROOT=args.data_path + 'slam/tum_scenes/', resolution=(518, 392), kf_every=2),
    'eth3d': Tum(ROOT=args.data_path + 'slam/eth_slam/', resolution=(518, 392), kf_every=2),
}

for demo_name, data in eval_datasets_all.items():
    print(f"Evaluating on {demo_name} dataset")
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    scene_results = {}

    for i in range(args.num_iters):
        iter_results_path = os.path.join(args.results_path, demo_name, f"iter_{i}")
        os.makedirs(iter_results_path, exist_ok=True)

        iter_ate = []
        iter_ate_kf = []
        iter_fps = []

        for _, batch in enumerate(dataloader):
            views, views_all = batch
            img_paths_all = [v['im_path'][0] for v in views] if 'im_path' in views[0] else None
            num_frames = views_all['images'].shape[1]

            scene_name = views[0]['label'][0].rsplit('/')[0]
            print(f"Processing scene {scene_name} with {num_frames} frames...")

            images = views_all['images']  # (1, T, 3, H, W), in [-1, 1] range
            poses_gt = views_all['camera_pose'][0].cpu().numpy()

            start_time = time.time()
            res = pipeline.run(images, poses_gt=poses_gt)
            end_time = time.time()
            fps = num_frames / (end_time - start_time)

            poses_pred = res.poses
            pts_pred = res.pts
            conf = res.conf
            kf_idx = res.kf_idx
            pts_mask, sky_mask = get_pts_mask(pts_pred, views_all['images'], conf, 
                                              conf_threshold=args.conf_threshold,
                                              segformer_path='../checkpoints/segformer.b0.512x512.ade.160k.pth')

            try:
                ape_stat, info = evaluate_evo(poses_gt, poses_pred, iter_results_path, f"scene_{scene_name}", monocular=True)
                iter_ate.append(ape_stat)
                iter_fps.append(fps)

                # Track per-scene results globally
                if scene_name not in scene_results:
                    scene_results[scene_name] = {'ape': [], 'ape_kf': [], 'fps': []}
                scene_results[scene_name]['ape'].append(ape_stat)
                scene_results[scene_name]['fps'].append(fps)

                ape_stat_kf = None
                if len(kf_idx) > 1:
                    poses_kf = poses_pred[kf_idx.cpu().numpy()]
                    poses_gt_kf = poses_gt[kf_idx.cpu().numpy()]
                    ape_stat_kf, info_kf = evaluate_evo(poses_gt_kf, poses_kf, iter_results_path, f"scene_{scene_name}_kf", monocular=True)
                    iter_ate_kf.append(ape_stat_kf)
                    scene_results[scene_name]['ape_kf'].append(ape_stat_kf)

                # Save per-scene result for this iteration
                with open(os.path.join(iter_results_path, "scenes_ape.txt"), 'a') as f:
                    if ape_stat_kf is not None:
                        f.write(f"Scene {scene_name}: APE = {ape_stat}, Keyframe APE = {ape_stat_kf}, fps = {fps:.2f}\n")
                    else:
                        f.write(f"Scene {scene_name}: APE = {ape_stat}, fps = {fps:.2f}\n")

            except Exception as e:
                print(f"Error during evaluation for scene {scene_name}: {e}")

            # Save pts
            pts_kf = pts_pred[kf_idx].cpu().numpy().reshape(-1, 3)
            color_kf = (images[0, kf_idx].cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 3) + 1.0) / 2.0

            print(pts_pred.shape, pts_mask.shape, kf_idx.shape)
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
            o3d.io.write_point_cloud(os.path.join(iter_results_path, f"scene_{scene_name}_keyframe_points.ply"), pcd)

        # --- Iteration summary ---
        if iter_ate:
            with open(os.path.join(iter_results_path, "scenes_ape.txt"), 'a') as f:
                f.write(f"\nIter {i} Average APE: {np.mean(iter_ate):.4f}, Median: {np.median(iter_ate):.4f}\n")
            print(f"Iter {i} Average APE: {np.mean(iter_ate):.4f}, Median: {np.median(iter_ate):.4f}")
        if iter_ate_kf:
            with open(os.path.join(iter_results_path, "scenes_ape.txt"), 'a') as f:
                f.write(f"Iter {i} Average KF APE: {np.mean(iter_ate_kf):.4f}, Median: {np.median(iter_ate_kf):.4f}\n")
        if iter_fps:
            with open(os.path.join(iter_results_path, "scenes_ape.txt"), 'a') as f:
                f.write(f"Iter {i} Average FPS: {np.mean(iter_fps):.2f}\n")

    # --- Dataset summary across all iterations ---
    dataset_results_path = os.path.join(args.results_path, demo_name)
    print(f"\nWriting {demo_name} summary...")
    with open(os.path.join(dataset_results_path, f"results_{demo_name}.txt"), 'w') as f:
        all_apes = []
        all_kf_apes = []
        all_fps = []

        for scene in sorted(scene_results.keys()):
            r = scene_results[scene]
            mean_ape = np.mean(r['ape'])
            mean_kf = np.mean(r['ape_kf']) if r['ape_kf'] else float('nan')
            mean_fps = np.mean(r['fps'])
            f.write(f"Scene {scene}: APE = {mean_ape:.4f}, KF APE = {mean_kf:.4f}, FPS = {mean_fps:.2f}  (over {len(r['ape'])} runs)\n")
            all_apes.extend(r['ape'])
            all_kf_apes.extend(r['ape_kf'])
            all_fps.extend(r['fps'])

        f.write(f"\nOverall Average APE: {np.mean(all_apes):.4f}\n")
        if all_kf_apes:
            f.write(f"Overall Average KF APE: {np.mean(all_kf_apes):.4f}\n")
        f.write(f"Overall Average FPS: {np.mean(all_fps):.2f}\n")
    print(f"*** {demo_name} summary saved to {os.path.join(dataset_results_path, f'results_{demo_name}.txt')}")