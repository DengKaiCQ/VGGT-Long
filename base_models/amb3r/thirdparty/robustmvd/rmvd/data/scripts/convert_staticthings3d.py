#!/usr/bin/env python3

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/scripts/convert_staticthings3d.py

import os
import os.path as osp
import sys
import re
import argparse

import numpy as np


def transform_from_rot_trans(R, t):
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1])).astype(np.float32)


def invert_transform(T):
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    R_inv = R.T
    t_inv = np.dot(-R.T, t)
    return transform_from_rot_trans(R_inv, t_inv)


def link_file(in_path, out_path, split, seqtype, seq, dtype, cam, frame_num, ext):
    seq = str(seq).zfill(4)
    frame_num = str(frame_num).zfill(4)

    p = osp.join(out_path, split)
    os.makedirs(p, exist_ok=True)
    p = osp.join(p, seqtype)
    os.makedirs(p, exist_ok=True)
    p = osp.join(p, seq)
    os.makedirs(p, exist_ok=True)
    p = osp.join(p, dtype)
    os.makedirs(p, exist_ok=True)
    p = osp.join(p, cam)
    os.makedirs(p, exist_ok=True)

    p = osp.join(p, frame_num+"."+ext)
    if not osp.isfile(p):
        print("Linking {} to {}".format(in_path, p))
        os.symlink(in_path, p)


def link_images(in_path, out_path):
    p = osp.join(in_path, 'frames_cleanpass')
    for split in os.listdir(p):
        p1 = osp.join(p, split)
        for seqtype in os.listdir(p1):
            p2 = osp.join(p1, seqtype)
            for seq in os.listdir(p2):
                p3 = osp.join(p2, seq)
                for cam in os.listdir(p3):
                    p4 = osp.join(p3, cam)
                    for frame in os.listdir(p4):
                        p5 = osp.join(p4, frame)
                        frame_num, ext = osp.splitext(frame)
                        frame_num = int(frame_num)
                        link_file(in_path=p5, out_path=out_path, split=split, seqtype=seqtype, seq=seq,
                                  dtype="frames_cleanpass", cam=cam, frame_num=frame_num, ext="png")
                        
    p = osp.join(in_path, 'frames_finalpass')
    for split in os.listdir(p):
        p1 = osp.join(p, split)
        for seqtype in os.listdir(p1):
            p2 = osp.join(p1, seqtype)
            for seq in os.listdir(p2):
                p3 = osp.join(p2, seq)
                for cam in os.listdir(p3):
                    p4 = osp.join(p3, cam)
                    for frame in os.listdir(p4):
                        p5 = osp.join(p4, frame)
                        frame_num, ext = osp.splitext(frame)
                        frame_num = int(frame_num)
                        link_file(in_path=p5, out_path=out_path, split=split, seqtype=seqtype, seq=seq,
                                  dtype="frames_finalpass", cam=cam, frame_num=frame_num, ext="png")
                        
                        
def link_depths(in_path, out_path):
    p = osp.join(in_path, 'depths')
    for split in os.listdir(p):
        p1 = osp.join(p, split)
        for seqtype in os.listdir(p1):
            p2 = osp.join(p1, seqtype)
            for seq in os.listdir(p2):
                p3 = osp.join(p2, seq)
                for cam in os.listdir(p3):
                    p4 = osp.join(p3, cam)
                    for frame in os.listdir(p4):
                        p5 = osp.join(p4, frame)
                        frame_num, ext = osp.splitext(frame)
                        frame_num = int(frame_num)
                        link_file(in_path=p5, out_path=out_path, split=split, seqtype=seqtype, seq=seq,
                                  dtype="depths", cam=cam, frame_num=frame_num, ext="float3")
                        
                        
def link_poses(in_path, out_path):
    p = osp.join(in_path, 'poses')
    for split in os.listdir(p):
        p1 = osp.join(p, split)
        for seqtype in os.listdir(p1):
            p2 = osp.join(p1, seqtype)
            for seq in os.listdir(p2):
                p3 = osp.join(p2, seq)
                for cam in os.listdir(p3):
                    p4 = osp.join(p3, cam)
                    for frame in os.listdir(p4):
                        p5 = osp.join(p4, frame)
                        frame_num, ext = osp.splitext(frame)
                        frame_num = int(frame_num)
                        link_file(in_path=p5, out_path=out_path, split=split, seqtype=seqtype, seq=seq,
                                  dtype="poses", cam=cam, frame_num=frame_num, ext="float3")
                        
                        
def link_intrinsics(in_path, out_path):
    p = osp.join(in_path, 'intrinsics')
    for split in os.listdir(p):
        p1 = osp.join(p, split)
        for seqtype in os.listdir(p1):
            p2 = osp.join(p1, seqtype)
            for seq in os.listdir(p2):
                p3 = osp.join(p2, seq)
                for cam in ['left', 'right']:
                    p4 = p3
                    for frame in os.listdir(p4):
                        p5 = osp.join(p4, frame)
                        frame_num, ext = osp.splitext(frame)
                        frame_num = int(frame_num)
                        link_file(in_path=p5, out_path=out_path, split=split, seqtype=seqtype, seq=seq,
                                  dtype="intrinsics", cam=cam, frame_num=frame_num, ext="float3")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    link_images(args.in_path, args.out_path)
    link_depths(args.in_path, args.out_path)
    link_poses(args.in_path, args.out_path)
    link_intrinsics(args.in_path, args.out_path)
    print("Done")
