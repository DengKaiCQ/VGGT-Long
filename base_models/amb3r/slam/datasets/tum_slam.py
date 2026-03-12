# References:
#   https://github.com/HengyiWang/amb3r/blob/main/slam/datasets/tum_slam.py

import os
import re
import cv2
import sys
import torch
import PIL
import numpy as np
import os.path as osp
from collections import deque
from scipy.spatial.transform import Rotation

# Import the standard PyTorch Dataset class
from torch.utils.data import Dataset
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

from dust3r.utils.image import imread_cv2
import dust3r.datasets.utils.cropping as cropping


class Tum(Dataset):
    def __init__(self, 
                 split='train', resolution=None,
                 full_video=True, 
                 skip=None,
                 use_calib=False,
                 eth_bench=False,
                 kf_every=1, ROOT=None, **kwargs):
                
        self.ROOT = ROOT
        self.split = split
        self.resolution = resolution
        self.full_video = full_video
        self.kf_every = kf_every
        self.skip = skip
        self.use_calib = use_calib
        self.eth_bench = eth_bench

         # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        
        views = self._get_views(idx, resolution=self.resolution, rng=rng)
        
        images = [torch.from_numpy(np.asarray(v['img'])).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0 for v in views]
        camera_poses = [torch.from_numpy(v['camera_pose']) for v in views]
        camera_intrinsics = [torch.from_numpy(v['camera_intrinsics']) for v in views]

        images_tensor = torch.stack(images)
        poses_tensor = torch.stack(camera_poses)
        intrinsics_tensor = torch.stack(camera_intrinsics)

        views_all = {
            'images': images_tensor,
            'camera_pose': poses_tensor,
            'camera_intrinsics': intrinsics_tensor,
        }
        
        return views, views_all
    
    def load_all_scenes(self, base_dir):
        self.scene_list = os.listdir(base_dir)

        if self.eth_bench:
            self.scene_list = [
                'cables_1',
                'camera_shake_1',
                'einstein_1',
                'plant_1',
                'plant_2',
                'sofa_1',
                'table_3',
                'table_7'
            ]          
        

        if self.skip is not None and self.skip >= 1:
            self.scene_list = self.scene_list[self.skip:]

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
    

    def associate_frames_nodepth(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """
        Associates images with depth and pose data, always returning a triplet
        (i, j, k) corresponding to (image, depth, pose) indices.

        If a data stream (e.g., depth) is not provided, its corresponding index in
        the triplet (j) will be None.

        An association is only created if all *available* data streams have a
        timestamp within the max_dt of the image timestamp.
        """
        associations = []
        for i, t_img in enumerate(tstamp_image):

            j, k = None, None
            valid_match = True

            if tstamp_depth is not None:
                j_candidate = np.argmin(np.abs(tstamp_depth - t_img))
                if np.abs(tstamp_depth[j_candidate] - t_img) < max_dt:
                    j = j_candidate
                else:
                    valid_match = False

            if tstamp_pose is not None:
                k_candidate = np.argmin(np.abs(tstamp_pose - t_img))
                if np.abs(tstamp_pose[k_candidate] - t_img) < max_dt:
                    k = k_candidate
                else:
                    valid_match = False

            if valid_match and (tstamp_depth is not None or tstamp_pose is not None):
                associations.append((i, j, k))

        return associations
    
    
    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations
    
    
    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data


    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)

        if osp.exists(depth_list):
            depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        if osp.exists(depth_list):
            tstamp_depth = depth_data[:, 0].astype(np.float64)
        else:
            tstamp_depth = None
        tstamp_pose = pose_data[:, 0].astype(np.float64)

        if tstamp_depth is None:
            associations = self.associate_frames_nodepth(
                tstamp_image, tstamp_depth, tstamp_pose)
        else:
            associations = self.associate_frames(
                tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            c2w = c2w
            poses += [c2w]

        return images, depths, poses


    def _get_views(self, idx, resolution, rng, num_frames=None):

        image_names, depth_names, poses = self.loadtum(osp.join(self.ROOT, self.scene_list[idx]))

        scene_id = self.scene_list[idx]
        num_images = len(image_names)

        if self.full_video:
            img_idxs = list(range(0, num_images, self.kf_every))
        else:
            raise NotImplementedError("Use full video mode for now.")

        datapath = osp.join(self.ROOT, self.scene_list[idx])
        kf_filepath = osp.join(datapath, f'kf_{self.kf_every}.txt')
        with open(kf_filepath, 'w') as f:
            for im_idx in img_idxs:
                filename = osp.basename(image_names[im_idx])
                f.write(f'{filename}\n')

        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            rgb_image = imread_cv2(image_names[im_idx])
            depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)

            if self.use_calib:
                match = re.search(r"freiburg(\d+)", scene_id)
                idx = int(match.group(1))
                if idx == 1:
                    fx, fy = 517.3, 516.5
                    cx, cy = 318.6, 255.3
                    
                if idx == 2:
                    fx, fy = 520.9, 521.0
                    cx, cy = 325.1, 249.7
                if idx == 3:
                    fx, fy = 535.4, 539.2
                    cx, cy = 320.1, 247.6

            else:
                fx, fy = 535.4, 539.2
                cx, cy = rgb_image.shape[1]//2, rgb_image.shape[0]//2
            
            intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            camera_pose = poses[im_idx].astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=scene_id)
            
            views.append(dict(
                img=rgb_image,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='tum',
                label=scene_id,
                instance=osp.split(image_names[im_idx])[-1],
            ))
        return views


    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, disable_crop=False):
            """ This function:
                - first downsizes the image with LANCZOS inteprolation,
                which is better than bilinear interpolation in
            """
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            

            if disable_crop:
                W_orig, H_orig = image.size
                # Make a mutable copy of the target resolution
                target_resolution = list(resolution)

                # Transpose the target resolution for portrait-oriented images
                # This logic is preserved from the original function
                assert target_resolution[0] >= target_resolution[1]
                if H_orig > 1.1 * W_orig:
                    # Image is in portrait mode
                    target_resolution = target_resolution[::-1]
                elif 0.9 < H_orig / W_orig < 1.1 and target_resolution[0] != target_resolution[1]:
                    # Image is square, so we choose orientation randomly
                    if rng is not None and rng.integers(2):
                        target_resolution = target_resolution[::-1]

                target_W, target_H = target_resolution
                
                # Directly resize the image using high-quality Lanczos interpolation
                image = image.resize((target_W, target_H), resample=PIL.Image.LANCZOS)
                
                # Resize the depth map. NEAREST is used to prevent interpolation
                # of depth values, which would create incorrect data.
                if isinstance(depthmap, np.ndarray):
                    depthmap_pil = PIL.Image.fromarray(depthmap)
                    depthmap_pil = depthmap_pil.resize((target_W, target_H), resample=PIL.Image.NEAREST)
                    depthmap = np.array(depthmap_pil)
                elif isinstance(depthmap, PIL.Image.Image):
                    depthmap = depthmap.resize((target_W, target_H), resample=PIL.Image.NEAREST)

                # Update camera intrinsics based on the scaling factors
                sx = target_W / W_orig
                sy = target_H / H_orig
                
                intrinsics_new = intrinsics.copy()
                intrinsics_new[0, 0] *= sx  # fx' = fx * (w_new / w_old)
                intrinsics_new[1, 1] *= sy  # fy' = fy * (h_new / h_old)
                intrinsics_new[0, 2] *= sx  # cx' = cx * (w_new / w_old)
                intrinsics_new[1, 2] *= sy  # cy' = cy * (h_new / h_old)
                
                return image, depthmap, intrinsics_new

            # downscale with lanczos interpolation so that image.size == resolution
            # cropping centered on the principal point
            W, H = image.size
            cx, cy = intrinsics[:2, 2].round().astype(int)
            
            # calculate min distance to margin
            min_margin_x = min(cx, W-cx)
            min_margin_y = min(cy, H-cy)
            assert min_margin_x > W/5, f'Bad principal point in view={info}'
            assert min_margin_y > H/5, f'Bad principal point in view={info}'
            
            ## Center crop
            # Crop on the principal point, make it always centered
            # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
            l, t = cx - min_margin_x, cy - min_margin_y
            r, b = cx + min_margin_x, cy + min_margin_y
            crop_bbox = (l, t, r, b)

            image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

            # transpose the resolution if necessary
            W, H = image.size  # new size
            assert resolution[0] >= resolution[1]
            if H > 1.1*W:
                # image is portrait mode
                resolution = resolution[::-1]
            elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
                # image is square, so we chose (portrait, landscape) randomly
                if rng.integers(2):
                    resolution = resolution[::-1]

            # high-quality Lanczos down-scaling
            target_resolution = np.array(resolution)
            
            ## Recale with max factor, so  one of width or height might be larger than target_resolution
            image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

            # actual cropping (if necessary) with bilinear interpolation
            intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
            crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
            image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

            return np.array(image), depthmap, intrinsics2



