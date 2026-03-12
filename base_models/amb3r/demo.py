# References:
#   https://github.com/HengyiWang/amb3r/blob/main/demo.py

import torch
import argparse
import open3d as o3d

from amb3r.model import AMB3R
from amb3r.datasets import Demo
from amb3r.tools.vis import interactive_pcd_viewer

from torch.utils.data import DataLoader


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./assets/demo_brunch/")
    return parser



def main():
    parser = get_args_parser()
    args = parser.parse_args()
    model = AMB3R()

    ckpt_path = './checkpoints/amb3r.pt'
    model.load_weights(ckpt_path)
    model.cuda()

    data = Demo(ROOT=args.data_path, resolution=(518, 392), num_seq=1, full_video=True, kf_every=1, disable_crop=False, max_images=40)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)

    batch = dataloader.__iter__().__next__()

    _, views_all = batch


    for key in views_all.keys():
        views_all[key] = views_all[key].cuda()


    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            res = model(views_all)


    # Extract raw data (keep unfiltered for interactive thresholding)
    pts = res[-1]['world_points'].cpu().numpy().reshape(-1, 3)
    conf = res[-1]['world_points_conf'].cpu().numpy().reshape(-1)
    conf_sig = (conf-1)/ conf
    color = res[-1]['images'].permute(0, 1, 3, 4, 2).reshape(-1, 3).cpu().numpy()
    c2w = res[-1]['pose'].cpu().numpy()

    pts0 = res[0]['world_points'].cpu().numpy().reshape(-1, 3)
    conf0 = res[0]['world_points_conf'].cpu().numpy().reshape(-1)
    conf_sig0 = (conf0-1)/ conf0
    color0 = res[0]['images'].permute(0, 1, 3, 4, 2).reshape(-1, 3).cpu().numpy()
    c2w0 = res[0]['pose'].cpu().numpy()

    # Full-resolution tensors for edge mask computation (squeeze batch dim)
    pts_full = res[-1]['world_points'][0].cpu().numpy()     # (T, H, W, 3)
    conf_full = ((res[-1]['world_points_conf'] - 1) / res[-1]['world_points_conf'])[0].cpu().numpy()  # (T, H, W)
    images_tensor = res[-1]['images']  # (B, T, C, H, W) on GPU

    print("Launching interactive viewer")
    # Launch interactive viewer
    interactive_pcd_viewer(
        pts, color, conf_sig,
        pts_raw_0=pts0, colors_raw_0=color0, conf_raw_0=conf_sig0,
        c2w=c2w, c2w_0=c2w0,
        images=images_tensor,
        pts_full=pts_full,
        conf_full=conf_full,
        initial_conf_thresh=0.0,
    )
    

if __name__ == "__main__":
    main()
