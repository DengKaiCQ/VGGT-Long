# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/bonn.py

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


class Bonn(BaseManyViewDataset):
    def __init__(self, num_seq=1, full_video=False, 
                 kf_every=1, *args, ROOT, **kwargs):
        
        ROOT = osp.join(ROOT, "rgbd_bonn_dataset")
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
        self.scene_list = os.listdir(base_dir)
        
    def depth_read_bonn(self, filename: str):
        depth_png = np.asarray(Image.open(filename))
        assert np.max(depth_png) > 255
        depth = depth_png.astype(np.float64) / 5000.0
        depth[depth_png == 0] = -1.0
        return depth.astype(np.float32)
    
    def convert_poses(
        self,
        poses,
    ):
        if len(poses.shape) == 1:
            poses = poses[None, :]
        b, _ = poses.shape

        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics
    

    def _get_views(self, idx, resolution, num_frames, rng):

        scene_id = self.scene_list[idx]


        image_path = osp.join(self.ROOT, scene_id, "rgb_110")
        depth_path = osp.join(self.ROOT, scene_id, "depth_110")


        image_names = sorted(os.listdir(image_path))
        depth_names = sorted(os.listdir(depth_path))

        num_images = len(image_names)

        if self.full_video:
            img_idxs = list(range(0, num_images, self.kf_every))
        else:
            raise NotImplementedError("Only full_video mode is supported for Bonn dataset.")


        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()


            rgb_image = imread_cv2(osp.join(image_path, image_names[im_idx]))
            depthmap = self.depth_read_bonn(osp.join(depth_path, depth_names[im_idx]))

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
                dataset='bonn',
                label=osp.join(scene_id, image_names[im_idx]),
                instance=osp.split(image_names[im_idx])[-1],
            ))
        return views







                    



