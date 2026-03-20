# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/eval_mvrecon.py

import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

from amb3r.model import AMB3R
from amb3r.datasets import SevenScenes, Eth3d, Dtu
from tools.pts_eval import accuracy, completion, compute_abs_rel
from vggt.train_utils.normalization import normalize_camera_extrinsics_and_points_batch

from torch.utils.data import DataLoader


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../data/rmvd/")
    parser.add_argument('--results_path', type=str, default="./outputs/mvrecon")
    parser.add_argument('--pts_by_unprojection', type=bool, default=True)
    return parser

args = get_args_parser().parse_args()

ckpt_path = '../checkpoints/amb3r.pt'
model = AMB3R()
model.load_weights(ckpt_path)
model.cuda()


os.makedirs(args.results_path, exist_ok=True)

eval_datasets_all = {
    '7scenes': SevenScenes(split='test', ROOT=args.data_path + '7scenes', resolution=(518, 392), num_seq=1, full_video=True, kf_every=40),
    'eth3d': Eth3d(ROOT=args.data_path + 'eth3d', resolution=(518, 392), rmvd_split=True),
    'dtu': Dtu(ROOT=args.data_path + 'dtu_with_poses', resolution=(518, 392)),

}

for demo_name, data in eval_datasets_all.items():
    print(f"Evaluating on {demo_name} dataset")
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)


    acc_all = []
    acc_med_all = []
    comp_all = []
    comp_med_all = []

    abs_rel_all = []


    for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {demo_name}")):

        views, views_all = batch

        views_all['extrinsics'], _, views_all['pts3d'], views_all['depthmap'], views_all['scale'] = normalize_camera_extrinsics_and_points_batch(
                    extrinsics=views_all['extrinsics'],
                    cam_points=None,
                    world_points=views_all['pts3d'],
                    depths=views_all['depthmap'],
                    scale_by_points=False,
                    point_masks=views_all['valid_mask'],
                    pred_points=None
                )

        for key in views_all.keys():
            if key == 'images':
                views_all[key] = views_all[key].cuda()

                print(views_all[key].shape)


        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                res = model.run_amb3r_benchmark(views_all)

        if args.pts_by_unprojection:
            if 'pts3d_by_unprojection' not in res:
                print("Warning: pts3d_by_unprojection not in res, using world_points instead")
                pts =  res['world_points'].cpu()
            else:
                pts = res['pts3d_by_unprojection'].cpu()
        else:
            pts = res['world_points'].cpu()

        
        

        gt_pts = views_all['pts3d']
        valid_mask = views_all['valid_mask']


        pts_masked = pts[valid_mask> 0].reshape(-1, 3)
        pts_gt_masked = gt_pts[valid_mask> 0].reshape(-1, 3)

        del res, views_all


        pts_dist = pts_masked.norm(dim=-1)
        gt_dist = pts_gt_masked.norm(dim=-1)

        pts0_factor = torch.nanmedian(pts_dist, dim=0, keepdim=True).values
        gt_pts_factor = torch.nanmedian(gt_dist, dim=0, keepdim=True).values

        pts_masked = pts_masked / pts0_factor * gt_pts_factor


        abs_rel = compute_abs_rel(pts_masked.cpu().numpy(), pts_gt_masked.cpu().numpy())


        pts_masked = pts_masked.cpu().numpy()
        pts_gt_masked = pts_gt_masked.cpu().numpy()


        acc, acc_med = accuracy(pts_gt_masked, pts_masked)
        comp, comp_med = completion(pts_gt_masked, pts_masked)


        print(f'Seq {i} / {len(dataloader)}: Abs Rel: {abs_rel:.4f}, Accuracy: {acc:.4f}, {acc_med:.4f}, Completion: {comp:.4f}, {comp_med:.4f}')


        scene_id = views[0]['label'][0].rsplit('/', 1)[0]
        # Save acc and comp to a txt file
        with open(os.path.join(args.results_path, f'results_{demo_name}.txt'), 'a') as f:
            f.write(f'Seq {i}:{scene_id}: Abs Rel: {abs_rel:.4f}\n')
            f.write(f'Seq {i}:{scene_id}: Accuracy: {acc:.4f}, {acc_med:.4f}\n')
            f.write(f'Seq {i}:{scene_id}: Completion: {comp:.4f}, {comp_med:.4f}\n')
            

        acc_all.append(acc)
        acc_med_all.append(acc_med)
        comp_all.append(comp)
        comp_med_all.append(comp_med)
        abs_rel_all.append(abs_rel)

        print(f'Running Average Abs Rel: {np.mean(abs_rel_all):.4f}, Accuracy: {np.mean(acc_all):.4f}, {np.mean(acc_med_all):.4f}, Completion: {np.mean(comp_all):.4f}, {np.mean(comp_med_all):.4f}')


        del pts_masked, pts_gt_masked, pts, gt_pts, valid_mask
    

    print(f'Final Abs Rel: {np.mean(abs_rel_all):.4f}')
    print(f'Final Accuracy: {np.mean(acc_all):.4f}, {np.mean(acc_med_all):.4f}')
    print(f'Final Completion: {np.mean(comp_all):.4f}, {np.mean(comp_med_all):.4f}')

    if args.pts_by_unprojection:
        file_name = f'results_{demo_name}_pts_by_unprojection.txt'
    else:
        file_name = f'results_{demo_name}_world_points.txt'

    with open(os.path.join(args.results_path, file_name), 'a') as f:
        f.write(f'Final Abs Rel: {np.mean(abs_rel_all)}\n')
        f.write(f'Final Accuracy: {np.mean(acc_all)}, {np.mean(acc_med_all)}\n')
        f.write(f'Final Completion: {np.mean(comp_all)}, {np.mean(comp_med_all)}\n')


