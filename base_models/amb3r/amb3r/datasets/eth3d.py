# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/eth3d.py

import os
import sys
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque
from torch.utils.data import Dataset

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


class Eth3d(BaseManyViewDataset):
    def __init__(self, num_seq=1, 
                 test_id=None, full_video=False, 
                 kf_every=1, skip=0, rmvd_split=False,
                 *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.skip = skip
        self.rmvd_split = rmvd_split

         # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        if self.rmvd_split:
            json_path = osp.join(osp.dirname(osp.abspath(__file__)), "eth3d_rmvd.json")
            with open(json_path, "r") as f:
                self.scene_list = json.load(f)
        else:
            self.scene_list = sorted([f for f in os.listdir(base_dir)])[self.skip:]
    
    def _get_views(self, idx, resolution, num_frames, rng, attempts=0): 
        if self.rmvd_split:
            return self._get_views_rmvd(idx, resolution, num_frames, rng, attempts)
        else:
            return self._get_views_all(idx, resolution, num_frames, rng, attempts)

    def _get_views_rmvd(self, idx, resolution, num_frames, rng, attempts=0):

        scene_info = self.scene_list[idx]

        scene_name = scene_info['scene_name']
        image_names = scene_info['image_names']
        poses_all = scene_info['poses']
        intrinsics_all = scene_info['intrinsics']

        downsample_factor = 1



        views = []



        for i in range(len(poses_all)):
            img_name = image_names[i].split('/')[-1]
            img_path = osp.join(self.ROOT, scene_name, 'images/dslr_images', img_name)
            rgb_image = imread_cv2(img_path, cv2.IMREAD_COLOR)

            depthim_path = osp.join(self.ROOT, scene_name, 'ground_truth_depth/dslr_images', img_name)

            height, width = 4032, 6048
            depth = np.fromfile(depthim_path, dtype=np.float32).reshape(height, width)
            depthmap = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)
            depthmap_ori = depthmap.copy()




            camera_pose = np.array(poses_all[i], dtype=np.float32)
            camera_pose = np.linalg.inv(camera_pose)  # Invert the pose to get world to camera

            intri = np.array(intrinsics_all[i], dtype=np.float32)


            intri[0, :] /= downsample_factor
            intri[1, :] /= downsample_factor

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=img_path)
            

            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                depthmap_ori=depthmap_ori,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='eth3d',
                label=osp.join(scene_name, img_name),
                instance=osp.split(img_path)[1],
            )

                
            views.append(dict_info)
        
        return views

    def _get_views_all(self, idx, resolution, num_frames, rng, attempts=0):

        scene_id = self.scene_list[idx]

        # Load metadata
        depth_path = osp.join(self.ROOT, scene_id, 'ground_truth_depth', 'dslr_images')
        depth_names = sorted([f.replace('txt', 'JPG') for f in os.listdir(osp.join(self.ROOT, scene_id, 'poses'))])
        num_files = len(depth_names)


        pose_all = []
        intri_all = []
        name_all = []

        for file_name in depth_names:
            posepath = osp.join(self.ROOT, scene_id, 'poses', file_name.replace('JPG', 'txt'))
            camera_pose = np.loadtxt(posepath)

            intrinsic_path = osp.join(self.ROOT, scene_id, 'intrinsics', file_name.replace('JPG', 'txt'))
            intrinsic = np.loadtxt(intrinsic_path)

            pose_all.append(camera_pose)
            intri_all.append(intrinsic)
            name_all.append(file_name.replace('.JPG', ''))

        pose_all = np.stack(pose_all, axis=0)
        name_all = np.array(name_all)
        intri_all = np.stack(intri_all, axis=0)



        img_idx = range(0, num_files)
        imgs_idxs = self.sample_frame_idx(img_idx, rng, full_video=True, d_n='eth3d')

        imgs_idxs = deque(imgs_idxs)

        views = []


        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            downsample_factor = 4 # For faster loading and evaluation

            downsampled_rgb_dir = osp.join(self.ROOT, scene_id, 'images_downsampled_4x')
            downsampled_depth_dir = osp.join(self.ROOT, scene_id, 'depth_downsampled_4x')
            os.makedirs(downsampled_rgb_dir, exist_ok=True)
            os.makedirs(downsampled_depth_dir, exist_ok=True)

            full_height, full_width = 4032, 6048
            new_height, new_width = full_height // downsample_factor, full_width // downsample_factor

            downsampled_impath = osp.join(downsampled_rgb_dir, depth_names[im_idx])
            downsampled_depthim_path = osp.join(downsampled_depth_dir, depth_names[im_idx].replace('.JPG', '.npy'))

            if osp.exists(downsampled_impath) and osp.exists(downsampled_depthim_path):
                # If downsampled versions exist, load them
                rgb_image = cv2.imread(downsampled_impath, cv2.IMREAD_COLOR)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) # Ensure RGB format
                depthmap = np.load(downsampled_depthim_path)
                impath = downsampled_impath
            
            else:

                # Load image data
                impath = osp.join(self.ROOT, scene_id, 'images/dslr_images_undistorted', depth_names[im_idx])
                rgb_image = imread_cv2(impath, cv2.IMREAD_COLOR)

                depthim_path = osp.join(depth_path, depth_names[im_idx])

                height, width = 4032, 6048
                depth = np.fromfile(depthim_path, dtype=np.float32).reshape(height, width)
                depthmap = np.nan_to_num(depth, posinf=0., neginf=0., nan=0.)


                # Downsample the image and depth map
                rgb_image = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                depthmap = cv2.resize(depthmap, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                # Save the downsampled versions
                cv2.imwrite(downsampled_impath, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                np.save(downsampled_depthim_path, depthmap)


            

            imname = depth_names[im_idx].replace('npy', 'jpg')

            img_id = np.where(name_all == imname.replace('.JPG', ''))[0][0]  # Find the index of imname

            camera_pose = pose_all[img_id].astype(np.float32)
            camera_pose = np.linalg.inv(camera_pose)  # Invert the pose to get world to camera
            intri = intri_all[img_id].astype(np.float32)

            intri[0, :] /= downsample_factor
            intri[1, :] /= downsample_factor

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=impath)
            

            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='eth3d',
                label=osp.join(scene_id, str(im_idx)),
                instance=osp.split(impath)[1],
            )

                
            views.append(dict_info)
        
        return views

   