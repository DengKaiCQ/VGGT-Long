# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/seven_scenes.py

import os
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


class SevenScenes(BaseManyViewDataset):
    def __init__(self, num_seq=1, num_frames=5, 
                 test_id=None, full_video=True, 
                 seq_id=None, slam=False, skip=None,
                 kf_every=1, *args, ROOT, **kwargs):
        
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.slam = slam
        self.skip = skip

         # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq
    
    def load_all_scenes(self, base_dir):

        if self.slam:
            # All seq-01
            self.scene_list = ['chess/seq-01', 'fire/seq-01', 'heads/seq-01',
                               'office/seq-01', 'pumpkin/seq-01', 'redkitchen/seq-01',
                               'stairs/seq-01']
            print(f"SLAM mode: Found {len(self.scene_list)} sequences in split {self.split}")

            if self.skip is not None:
                self.scene_list = self.scene_list[self.skip:]
            return
            
        scenes = os.listdir(base_dir)
        file_split = {'train': 'TrainSplit.txt', 'test': 'TestSplit.txt'}[self.split]
        
        self.scene_list = []
        for scene in scenes:
            if self.test_id is not None and scene != self.test_id:
                continue
            # read file split
            with open(osp.join(base_dir, scene, file_split)) as f:
                seq_ids = f.read().splitlines()
                
                
                for seq_id in seq_ids:
                    num_part = ''.join(filter(str.isdigit, seq_id))
                    seq_id = f'seq-{num_part.zfill(2)}'
                    if self.seq_id is not None and seq_id != self.seq_id:
                        continue
                    self.scene_list.append(f"{scene}/{seq_id}")
        
        
        print(f"Found {len(self.scene_list)} sequences in split {self.split}")
    

    def _get_views(self, idx, resolution, num_frames, rng):


        scene_id = self.scene_list[idx // self.num_seq]

        data_path = osp.join(self.ROOT, scene_id)
        num_files = len([name for name in os.listdir(data_path) if 'color' in name])
        img_idxs = [f'{i:06d}' for i in range(num_files)]
        img_idxs = self.sample_frame_idx(img_idxs, rng, full_video=self.full_video)
        

        fx, fy, cx, cy = 525, 525, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(self.ROOT, scene_id, f'frame-{im_idx}.color.png')
            depthpath = osp.join(self.ROOT, scene_id, f'frame-{im_idx}.depth.proj.png')

            sfm_pose_path = osp.join(self.ROOT, scene_id, 'sfm_poses', f'frame-{im_idx}.txt')
            if osp.exists(sfm_pose_path):
                posepath = sfm_pose_path
            else:
                posepath = osp.join(self.ROOT, scene_id, f'frame-{im_idx}.pose.txt')

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap[depthmap==65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap>10] = 0
            depthmap[depthmap<1e-3] = 0

            
            camera_pose = np.loadtxt(posepath).astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath)
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='7scenes',
                label=osp.join(scene_id, im_idx),
                instance=osp.split(impath)[1],
            ))
        return views

