# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/dtu.py

import os
import re
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque
from torch.utils.data import Dataset

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


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
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    return data

        

class Dtu(BaseManyViewDataset):
    def __init__(self, num_seq=1, 
                 test_id=None, full_video=False, 
                 sequential=False,
                 kf_every=1, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.sequential = sequential

         # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):

        json_path = osp.join(osp.dirname(osp.abspath(__file__)), "dtu_rmvd.json")
        with open(json_path, "r") as f:
            self.scene_list = json.load(f)
    
    def _get_views(self, idx, resolution, num_frames, rng, attempts=0): 

        scene_info = self.scene_list[idx]

        scene_name = scene_info['scene_name']
        image_names = scene_info['image_names']
        poses_all = scene_info['poses']
        intrinsics_all = scene_info['intrinsics']

        if self.sequential:
            # Sort all list by image names
            sorted_indices = sorted(range(len(image_names)), key=lambda i: image_names[i])
            image_names = [image_names[i] for i in sorted_indices]
            poses_all = [poses_all[i] for i in sorted_indices]
            intrinsics_all = [intrinsics_all[i] for i in sorted_indices]






        views = []


        for i in range(len(poses_all)):
            img_name = image_names[i]
            img_path = osp.join(self.ROOT, scene_name, img_name)
            rgb_image = imread_cv2(img_path, cv2.IMREAD_COLOR)

            depthim_path = osp.join(self.ROOT, scene_name, img_name.replace('images', 'gt_depths').replace('.png', '.pfm'))
            depth = readPFM(depthim_path) / 1000.0  # Convert mm to m
            depthmap = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
            depthmap_ori = depthmap.copy()




            camera_pose = np.array(poses_all[i], dtype=np.float32)
            camera_pose = np.linalg.inv(camera_pose)  # Invert the pose to get world to camera

            intri = np.array(intrinsics_all[i], dtype=np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=img_path)
            
            # Check if the image is valid
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found")
                    continue
                else:
                    if attempts >= 5:
                        new_idx = rng.integers(0, self.__len__()-1)
                        return self._get_views(new_idx, resolution, num_frames, rng)
                    return self._get_views(idx, resolution, num_frames, rng, attempts+1)
            

            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                depthmap_ori=depthmap_ori,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='dtu',
                label=osp.join(scene_name, img_name),
                instance=osp.split(img_path)[1],
            )

                
            views.append(dict_info)
        
        return views

    