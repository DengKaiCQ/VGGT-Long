# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/rel10k.py

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



class Re10kAMB3R(BaseManyViewDataset):
    """
    Dataset for Re10K amb3r split.
    """
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list)
    
    def load_all_scenes(self, base_dir):
        self.scene_list = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        print(f"Found {len(self.scene_list)} scenes in {base_dir}")
        
    def convert_poses(self, poses):
        if len(poses.shape) == 1:
            poses = poses[None, :]
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c, intrinsics
    
    def load_metadata(self, example_path):
        with open(example_path, "r") as f:
            lines = f.read().splitlines()

        url = lines[0]

        timestamps = []
        cameras = []

        for line in lines[1:]:
            timestamp, *camera = line.split(" ")
            timestamps.append(int(timestamp))
            cameras.append(np.fromstring(",".join(camera), sep=","))

        timestamps = torch.tensor(timestamps, dtype=torch.int64)
        cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

        return {
            "url": url,
            "timestamps": timestamps,
            "cameras": cameras,
        }

    
    def _get_views(self, idx, resolution, num_frames, rng):

        scene_id = self.scene_list[idx]

        scene_path = osp.join(self.ROOT, scene_id)
        meta_data_path = osp.join(self.ROOT, scene_id + '.txt')
        metadata = self.load_metadata(meta_data_path)

        camera_info = metadata['cameras']
        timestamps = metadata['timestamps']

        # Images are already prefixed with order index: 0_timestamp.png, 1_timestamp.png, ...
        # Sort by filename to get them in the correct sampling order
        image_files = sorted(os.listdir(scene_path))

        views = []

        for im_idx, img_file in enumerate(image_files):
            rgb_image = imread_cv2(osp.join(scene_path, img_file))

            # Extract timestamp from filename: "0_timestamp.png" -> timestamp
            basename = osp.splitext(img_file)[0]
            timestamp_str = basename.split('_', 1)[1]  # Remove order prefix
            timestamp_from_file = int(timestamp_str)

            metadata_idx = (timestamps == timestamp_from_file).nonzero(as_tuple=True)[0].item()
            cam_info = camera_info[metadata_idx]

            h, w, _ = rgb_image.shape

            depthmap = np.zeros_like(rgb_image[..., 0], dtype=np.float32)

            extrinsic, intrinsics_ = self.convert_poses(cam_info)

            extrinsic = extrinsic[0].numpy()
            intrinsics_ = intrinsics_[0].numpy()

            intrinsics_[0, :] *= w
            intrinsics_[1, :] *= h

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=scene_path)
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=np.linalg.inv(extrinsic),
                extrinsic=extrinsic,
                camera_intrinsics=intrinsics,
                dataset='re10k',
                label=osp.join(scene_id, str(im_idx)),
                instance=scene_id,
            ))
        return views
