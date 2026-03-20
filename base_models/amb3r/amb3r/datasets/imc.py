# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/imc.py

import os
import sys
import cv2
import numpy as np
import os.path as osp
from collections import deque

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset

class Imc(BaseManyViewDataset):
    def __init__(self, num_seq=1, scene_folder='test', 
                 skip=0, kf_every=1, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.scene_folder = scene_folder  # Store this to use for filtering
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.kf_every = kf_every
        self.skip = skip

        # load all scenes
        self.load_all_scenes(self.ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        all_items = sorted(os.listdir(base_dir))
        valid_scenes = []
        for d in all_items:
            scene_path = osp.join(base_dir, d)
            if osp.isdir(scene_path) and osp.exists(osp.join(scene_path, 'dense', 'images')):
                valid_scenes.append(d)
        
        if self.scene_folder == 'test':
            test_scenes = [
                "british_museum",
                "florence_cathedral_side",
                "lincoln_memorial_statue",
                "milan_cathedral",
                "mount_rushmore",
                "piazza_san_marco",
                "sagrada_familia",
                "st_pauls_cathedral"
            ]
            missing_scenes = [s for s in test_scenes if s not in valid_scenes]
            if missing_scenes:
                raise ValueError(f"Missing test scenes: {missing_scenes}")
            valid_scenes = [s for s in valid_scenes if s in test_scenes]
            
        self.scene_list = valid_scenes[self.skip:]

        print(f"Loaded {len(self.scene_list)} scenes from {self.scene_folder} split.")
    
    def _get_views(self, idx, resolution, num_frames, rng, attempts=0): 
        scene_id = self.scene_list[idx]
        image_dir = osp.join(self.ROOT, scene_id, 'dense', 'images')
        pose_dir = osp.join(self.ROOT, scene_id, 'dense', 'poses')

        # Pair images with their corresponding extracted pose files
        valid_frames = []
        for img_name in sorted(os.listdir(image_dir)):
            base_name = osp.splitext(img_name)[0]
            pose_path = osp.join(pose_dir, f"{base_name}.txt")
            if osp.exists(pose_path):
                valid_frames.append((img_name, pose_path))
                
        if len(valid_frames) == 0:
            raise ValueError(f"No valid frames found in scene {scene_id}")

        img_idx = range(len(valid_frames))
        imgs_idxs = self.sample_frame_idx(img_idx, rng, full_video=True, d_n='imc')
        imgs_idxs = deque(imgs_idxs)

        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            img_name, pose_path = valid_frames[im_idx]

            # 1. Load image data
            impath = osp.join(image_dir, img_name)
            rgb_image = imread_cv2(impath, cv2.IMREAD_COLOR)
            h, w = rgb_image.shape[:2]

            # 2. Load pose data
            camera_pose = np.loadtxt(pose_path).astype(np.float32)

            # 3. Create Placeholder Depth Map
            depthmap = np.zeros((h, w), dtype=np.float32)

            # 4. Create Placeholder Intrinsics
            fx = fy = max(h, w)
            cx, cy = w / 2.0, h / 2.0
            intri = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
           
            # 5. Crop & Resize
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=impath)
            
            # 6. Package View
            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='imc',
                label=osp.join(scene_id, str(im_idx)),
                instance=img_name,
            )

            views.append(dict_info)
        
        return views