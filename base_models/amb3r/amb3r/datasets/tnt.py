# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/tnt.py

import os
import cv2
import numpy as np
import os.path as osp
from collections import deque
from torch.utils.data import Dataset

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset
        

class Tnt(BaseManyViewDataset):
    def __init__(self, num_seq=1, num_frames=5, 
                 full_video=False, 
                 scene_folder='training',
                 kf_every=1, skip=0, *args, ROOT, **kwargs):
        self.ROOT = osp.join(ROOT, scene_folder)
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.full_video = full_video
        self.kf_every = kf_every
        self.skip = skip

         # load all scenes
        self.load_all_scenes(self.ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):

        self.scene_list = [f for f in os.listdir(base_dir)][self.skip:]
    
    def _get_views(self, idx, resolution, rng, attempts=0): 

        scene_id = self.scene_list[idx]

        scene_path = osp.join(self.ROOT, scene_id, 'dense', 'ibr3d_pw_0.50')
        rotation_all = np.load(osp.join(scene_path, 'Rs.npy')) # Bs, 3, 3

        num_files = len(rotation_all)

        translation_all = np.load(osp.join(scene_path, 'ts.npy')) # Bs, 3
        intrinsics_all = np.load(osp.join(scene_path, 'Ks.npy'))

        pose_all = np.concatenate([
            rotation_all.reshape(-1, 3, 3),
            translation_all.reshape(-1, 3, 1)
        ], axis=-1)

    

     
        if not self.full_video:
            raise NotImplementedError("Currently only support full video sampling for TNT dataset, please set full_video=True")

        else:
            img_idx = range(0, num_files)
            imgs_idxs = self.sample_frame_idx(img_idx, rng, full_video=self.full_video, d_n='wildrgbd')

        imgs_idxs = deque(imgs_idxs)

        views = []


        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            im_name = f'im_{im_idx:08d}.jpg'
            depth_name = f'dm_{im_idx:08d}.npy'

            # Load image data
            impath = osp.join(scene_path, 'image', im_name)
            rgb_image = imread_cv2(impath, cv2.IMREAD_COLOR)

            depthim_path = osp.join(scene_path, 'depth', depth_name)
            depthmap = np.load(depthim_path)


            camera_pose = pose_all[im_idx].astype(np.float32)
            camera_pose = np.concatenate([camera_pose, np.array([[0, 0, 0, 1]])], axis=0)  # Add the last row for homogeneous coordinates

            camera_pose = np.linalg.inv(camera_pose).astype(np.float32)  # Convert to camera to world pose
            intri = intrinsics_all[im_idx].astype(np.float32)
           

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=impath)
            
            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='tnt',
                label=osp.join(scene_id, str(im_idx)),
                instance=osp.split(impath)[1],
            )
      
            views.append(dict_info)
        
        return views 