#!/usr/bin/env python3

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/scripts/convert_flyingthings3d.py

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


def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    return data


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def copy_disparity(in_path, out_path, split, seqtype, seq, cam, frame_num):
    seq = str(seq).zfill(4)
    frame_num = str(frame_num).zfill(4)

    p = osp.join(out_path, split)
    os.makedirs(p, exist_ok=True)
    p = osp.join(p, seqtype)
    os.makedirs(p, exist_ok=True)
    p = osp.join(p, seq)
    os.makedirs(p, exist_ok=True)

    p_disp = osp.join(p, "disparities")
    os.makedirs(p_disp, exist_ok=True)
    p_disp = osp.join(p_disp, cam)
    os.makedirs(p_disp, exist_ok=True)

    p_disp = osp.join(p_disp, frame_num+".pfm")
    if cam == 'right':
        if not osp.isfile(p_disp):
            print("Linking {} to {}".format(in_path, p_disp))
            os.symlink(in_path, p_disp)
    elif cam == 'left':
        disp = readPFM(in_path)
        disp *= -1
        print("Writing {} with shape {}".format(p_disp, disp.shape))
        writePFM(p_disp, disp)


def copy_disparities(in_path, out_path):
    p = osp.join(in_path, 'disparity')
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
                        copy_disparity(in_path=p5, out_path=out_path, split=split, seqtype=seqtype, seq=seq,
                                       cam=cam, frame_num=frame_num)


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


def copy_calibration(in_path, out_path, split, seqtype, seq):
    seq = str(seq).zfill(4)

    p = osp.join(out_path, split)
    os.makedirs(p, exist_ok=True)
    p = osp.join(p, seqtype)
    os.makedirs(p, exist_ok=True)
    p = osp.join(p, seq)
    os.makedirs(p, exist_ok=True)

    po = osp.join(p, 'poses')
    os.makedirs(po, exist_ok=True)
    pleft = osp.join(po, 'left')
    os.makedirs(pleft, exist_ok=True)
    pright = osp.join(po, 'right')
    os.makedirs(pright, exist_ok=True)

    pk = osp.join(p, 'intrinsics')
    os.makedirs(pk, exist_ok=True)
    K_left = osp.join(pk, 'left')
    os.makedirs(K_left, exist_ok=True)
    K_right = osp.join(pk, 'right')
    os.makedirs(K_right, exist_ok=True)

    fx = 1050.
    fy = 1050.
    cx = 480
    cy = 270
    K = np.array([[fx, 0., cx], [0, fy, cy], [0, 0, 1]])

    with open(in_path, 'r') as f:
        poses = list(map(lambda x: x.strip(), f.readlines()))
    frames = poses[::4]
    poses_l = poses[1::4]
    # poses_r = poses[2::4]

    world_to_ref_transform = None
    L_to_center_transform = np.identity(4)
    L_to_center_transform[0][3] = +0.5
    R_to_center_transform = np.identity(4)
    R_to_center_transform[0][3] = -0.5
    rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]])

    for i in range(len(frames)):
        frame_num = str(int(frames[i][6:])).zfill(4)

        pose_l = poses_l[i].split(' ')[1:]
        world_to_L_transform = np.array([float(x) for x in pose_l]).reshape(4, 4)  # float64

        world_to_center_transform = world_to_L_transform.dot(L_to_center_transform)
        world_to_center_transform = np.dot(world_to_center_transform, rot)

        if world_to_ref_transform is None:
            world_to_ref_transform = world_to_center_transform.copy()  # use first center of the stereo pair as ref

        center_to_world_transform = invert_transform(world_to_center_transform)
        left_to_world_transform = L_to_center_transform.dot(center_to_world_transform)
        right_to_world_transform = R_to_center_transform.dot(center_to_world_transform)

        left_to_ref_transform = left_to_world_transform.dot(world_to_ref_transform)
        right_to_ref_transform = right_to_world_transform.dot(world_to_ref_transform)

        pleft_out = osp.join(pleft, frame_num+".npy")
        pright_out = osp.join(pright, frame_num+".npy")
        print("Writing left and right poses to \n\t{} and \n\t{}".format(pleft_out, pright_out))
        np.save(pleft_out, left_to_ref_transform)
        np.save(pright_out, right_to_ref_transform)

        K_left_out = osp.join(K_left, frame_num+".npy")
        K_right_out = osp.join(K_right, frame_num+".npy")
        print("Writing left and right intrinsics to \n\t{} and \n\t{}".format(K_left_out, K_right_out))
        np.save(K_left_out, K)
        np.save(K_right_out, K)


def copy_calibrations(in_path, out_path):
    p = osp.join(in_path, 'camera_data')
    for split in os.listdir(p):
        p1 = osp.join(p, split)
        for seqtype in os.listdir(p1):
            p2 = osp.join(p1, seqtype)
            for seq in os.listdir(p2):
                p3 = osp.join(p2, seq)
                p3 = osp.join(p3, 'camera_data.txt')
                copy_calibration(in_path=p3, out_path=out_path, split=split, seqtype=seqtype, seq=seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    link_images(args.in_path, args.out_path)
    copy_disparities(args.in_path, args.out_path)
    copy_calibrations(args.in_path, args.out_path)
    print("Done")
