# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/sintel.py

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

TAG_FLOAT = 202021.25

class Sintel(BaseManyViewDataset):
    def __init__(self, num_seq=1, full_video=False, 
                 kf_every=1, *args, ROOT, **kwargs):
        
        ROOT = osp.join(ROOT, "training")
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.full_video = full_video
        self.kf_every = kf_every

         # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq
    
    def load_all_scenes(self, base_dir):
        self.scene_list = os.listdir(osp.join(base_dir, "final"))
        
    def depth_read_sintel(self, filename):
        """Read depth data from file, return as numpy array."""
        f = open(filename, "rb")
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert (
            check == TAG_FLOAT
        ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
            TAG_FLOAT, check
        )
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert (
            width > 0 and height > 0 and size > 1 and size < 100000000
        ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
            width, height
        )
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
        return depth
    


    def _get_views(self, idx, resolution, num_frames, rng):

        scene_id = self.scene_list[idx]


        image_path = osp.join(self.ROOT, 'final', scene_id)
        depth_path = osp.join(self.ROOT, 'depth', scene_id)


        image_names = sorted(os.listdir(image_path))

        num_images = len(image_names)

        if self.full_video:
            img_idxs = list(range(0, num_images, self.kf_every))
        else:
            raise NotImplementedError("Only full_video mode is supported for Sintel dataset")



        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()


            rgb_image = imread_cv2(osp.join(image_path, image_names[im_idx]))
            depthmap = self.depth_read_sintel(osp.join(depth_path, image_names[im_idx].replace('.png', '.dpt')))

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
                dataset='sintel',
                label=osp.join(scene_id, image_names[im_idx]),
                instance=osp.split(image_names[im_idx])[-1],
            ))
        return views







                    



