#!/usr/bin/env python3
# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/scripts/convert_dtu.py

import os
import os.path as osp
import argparse

from tqdm import tqdm


def cp(a, b, verbose=True, followLinks=False):
    os.system('cp -r %s %s "%s" "%s"' % ('-v' if verbose else '', '-L' if followLinks else '', a, b))


def copy_rectified_images(in_base, out_base):
    in_base = osp.join(in_base, 'Rectified')
    scans = os.listdir(in_base)
    
    for scan in tqdm(scans, "Processed scans"):
        in_path = osp.join(in_base, scan)
        out_path = osp.join(out_base, scan)
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "images")
        os.makedirs(out_path, exist_ok=True)

        images = sorted([x for x in os.listdir(in_path) if x.endswith("_3_r5000.png")])

        for idx, image in enumerate(images):
            image_in = osp.join(in_path, image)
            image_out = osp.join(out_path, "{:08d}.png".format(idx))
            cp(image_in, image_out)


def copy_gt_depths(in_base, out_base):
    in_base = osp.join(in_base, 'dtu', 'Depths_raw')
    scans = os.listdir(in_base)

    for scan in tqdm(scans, "Processed scans"):
        in_path = osp.join(in_base, scan)
        out_path = osp.join(out_base, scan)
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "gt_depths")
        os.makedirs(out_path, exist_ok=True)

        depths = sorted([x for x in os.listdir(in_path) if x.endswith(".pfm")])

        for idx, depth in enumerate(depths):
            depth_in = osp.join(in_path, depth)
            depth_out = osp.join(out_path, "{:08d}.pfm".format(idx))
            cp(depth_in, depth_out)


def copy_points(in_base, out_base):
    in_base = osp.join(in_base, 'Points', 'stl')
    scans = [x for x in os.listdir(in_base) if x.endswith(".ply")]

    for scan in tqdm(scans, "Processed scans"):
        in_path = osp.join(in_base, scan)
        scan_id = int(scan[3:6])
        out_path = osp.join(out_base, "scan{}".format(scan_id))
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "scan.ply")
        cp(in_path, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    print("Copying rectified images:")
    copy_rectified_images(args.in_path, args.out_path)

    print("Copying GT depths:")
    copy_gt_depths(args.in_path, args.out_path)

    print("Copying points:")
    copy_points(args.in_path, args.out_path)

    print("Done")
