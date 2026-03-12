# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/eval_sfm.py

import os
import sys
import argparse
import open3d as o3d
import numpy as np
import torch
import collections

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.tools.pose_eval import get_results_from_camera_pose
from sfm.pipeline import AMB3R_SfM
from amb3r.model import AMB3R
from amb3r.tools.pts_vis import get_pts_mask
from amb3r.datasets import Eth3d, Tnt, Imc
from torch.utils.data import DataLoader


torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 for matmul on Ampere GPUs
torch.backends.cudnn.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/")
    parser.add_argument('--results_path', type=str, default="./outputs/sfm")
    parser.add_argument('--ckpt_path', type=str, default="../checkpoints/amb3r.pt")
    parser.add_argument('--num_iters', type=int, default=2)
    parser.add_argument('--target_point_count', type=int, default=3_000_000)
    return parser


args = get_args_parser().parse_args()

model = AMB3R()
model.load_weights(args.ckpt_path)
model.cuda()
model.eval()
pipeline = AMB3R_SfM(model, cfg_path='../sfm/sfm_config.yaml')

os.makedirs(args.results_path, exist_ok=True)

eval_datasets_all = {
    'eth3d': Eth3d(ROOT=args.data_path + 'rmvd/eth3d',
             resolution=(518, 392), kf_every=1, full_video=True),
    'tnt_training': Tnt(ROOT=args.data_path + 'sfm/tnt', resolution=(518, 392),
           kf_every=1, scene_folder='training', full_video=True),
    'tnt_intermediate': Tnt(ROOT=args.data_path + 'sfm/tnt', resolution=(518, 392),
           kf_every=1, scene_folder='intermediate', full_video=True),
    'tnt_advanced': Tnt(ROOT=args.data_path + 'sfm/tnt', resolution=(518, 392),
           kf_every=1, scene_folder='advanced', full_video=True),
    'imc': Imc(ROOT=args.data_path + 'sfm/imc', resolution=(518, 392),
           kf_every=1, scene_folder='test', full_video=True),
}

for demo_name, data in eval_datasets_all.items():
    print(f"Evaluating on {demo_name} dataset")
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    # Accumulate per-scene metrics across all iterations: {scene_id: {metric: [values]}}
    scene_metrics_all_iters = collections.defaultdict(lambda: collections.defaultdict(list))

    for i in range(args.num_iters):
        iter_results_path = os.path.join(args.results_path, demo_name, f"iter_{i}")
        os.makedirs(iter_results_path, exist_ok=True)
        total_metrics = collections.defaultdict(float)
        num_scenes_processed = 0

        for scene_idx, batch in enumerate(dataloader):
            views, views_all = batch
            img_paths_all = [v['im_path'][0] for v in views] if 'im_path' in views[0] else None
            num_frames = views_all['images'].shape[1]

            scene_id = views[0]['label'][0].rsplit('/')[0]

            print(f"Processing scene {scene_id} with {num_frames} frames.")

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    res = pipeline.run(views_all['images'], poses_gt=views_all['camera_pose'][0])

            poses_pred = res.poses
            pts_pred = res.pts
            conf = res.conf
            kf_idx = res.kf_idx
            poses_gt = views_all['camera_pose'][0].cpu()
            result, extri, gt_extri = get_results_from_camera_pose(poses_pred, poses_gt)
            print(f"Scene {scene_id} with results: {result}")

            with open(os.path.join(iter_results_path, f"scenes_metrics.txt"), 'a') as f:
                f.write(f"Scene {scene_id} with results: {result}\n")

            for key, value in result.items():
                total_metrics[key] += value
                scene_metrics_all_iters[scene_id][key].append(value)
            num_scenes_processed += 1

            # save the results
            pts_mask, sky_mask = get_pts_mask(pts_pred, views_all['images'], conf, 
                                              conf_threshold=0.01, segformer_path='../checkpoints/segformer.b0.512x512.ade.160k.pth')

            pts_masked = pts_pred[pts_mask]
            images = views_all['images'][0].permute(0, 2, 3, 1)
            color_masked = images[pts_mask]
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
            o3d.io.write_point_cloud(os.path.join(iter_results_path, f"scene_{scene_id}_points.ply"), pcd)

        if num_scenes_processed > 0:
            average_metrics = {key: total / num_scenes_processed for key, total in total_metrics.items()}

            print(f"\n=================================================")
            print(f"Iteration {i} Average Metrics ({num_scenes_processed} scenes):")
            print(average_metrics)
            print(f"=================================================\n")

            with open(os.path.join(iter_results_path, f"scenes_metrics.txt"), 'a') as f:
                f.write(f"\n=================================================\n")
                f.write(f"Iteration {i} Average Metrics ({num_scenes_processed} scenes):\n")
                f.write(f"{average_metrics}\n")
                f.write(f"=================================================\n\n")

    dataset_results_path = os.path.join(args.results_path, demo_name)
    overall_metrics = collections.defaultdict(float)
    num_scenes = len(scene_metrics_all_iters)

    with open(os.path.join(dataset_results_path, f"results_{demo_name}.txt"), 'w') as f:
        for scene_id, metrics in scene_metrics_all_iters.items():
            scene_avg = {key: np.mean(values) for key, values in metrics.items()}
            f.write(f"Scene {scene_id} (avg over {args.num_iters} iters): {scene_avg}\n")
            for key, value in scene_avg.items():
                overall_metrics[key] += value

        if num_scenes > 0:
            overall_avg = {key: total / num_scenes for key, total in overall_metrics.items()}
            f.write(f"\n=================================================\n")
            f.write(f"Overall Average ({num_scenes} scenes, {args.num_iters} iters):\n")
            f.write(f"{overall_avg}\n")
            f.write(f"=================================================\n")
            print(f"\n*** {demo_name} Overall Average ({num_scenes} scenes, {args.num_iters} iters): {overall_avg}")
