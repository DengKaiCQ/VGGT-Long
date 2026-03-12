# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/kitti.py

import os
import cv2
import torch
import numpy as np
import os.path as osp
from PIL import Image
from io import BytesIO
from collections import deque
from einops import rearrange, repeat


from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


class Kitti(BaseManyViewDataset):
    def __init__(self, num_seq=1, 
                 full_video=False, 
                 kf_every=1, *args, ROOT, **kwargs):
        
        ROOT = osp.join(ROOT, "depth_selection/val_selection_cropped/")
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.kf_every = kf_every
        self.full_video = full_video

         # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq
    
    def load_all_scenes(self, base_dir):
        self.scene_list = os.listdir(osp.join(base_dir, "image_gathered"))
        
    def depth_read_kitti(self, filename):
        img_pil = Image.open(filename)
        depth_png = np.array(img_pil, dtype=int)
        assert np.max(depth_png) > 255

        depth = depth_png.astype(float) / 256.0
        depth[depth_png == 0] = -1.0
        return depth.astype(np.float32)

    def _get_views(self, idx, resolution, num_frames, rng):

        scene_id = self.scene_list[idx]


        image_path = osp.join(self.ROOT, "image_gathered", scene_id)
        depth_path = osp.join(self.ROOT, "groundtruth_depth_gathered", scene_id)


        image_names = sorted(os.listdir(image_path))

        num_images = len(image_names)

        if self.full_video:
            img_idxs = list(range(0, num_images, self.kf_every))
        else:
            raise NotImplementedError("Only full_video mode is implemented for Kitti dataset.")


        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()


            rgb_image = imread_cv2(osp.join(image_path, image_names[im_idx]))
            depthmap = self.depth_read_kitti(osp.join(depth_path, image_names[im_idx]))

            depthmap_ori = depthmap.copy()


            cx, cy = rgb_image.shape[1]//2, rgb_image.shape[0]//2
            intrinsics_ = np.array([[1.0, 0, cx], [0, 1.0, cy], [0, 0, 1]], dtype=np.float32)


            camera_pose = np.eye(4, dtype=np.float32)

            
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=scene_id, disable_crop=True)
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                depthmap_ori=depthmap_ori,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='kitti',
                label=osp.join(scene_id, image_names[im_idx]),
                instance=osp.split(image_names[im_idx])[-1],
            ))
        return views

