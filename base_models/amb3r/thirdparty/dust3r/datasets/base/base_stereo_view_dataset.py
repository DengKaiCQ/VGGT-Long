# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# base class for implementing datasets
# --------------------------------------------------------
# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/dust3r/datasets/base/base_stereo_view_dataset.py

import PIL
import itertools
import numpy as np
import torch
import torch.nn.functional as F

from dust3r.datasets.base.easy_dataset import EasyDataset
from dust3r.datasets.utils.transforms import ImgNorm, ImgInvNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dust3r.datasets.utils.cropping as cropping
from scipy.spatial.transform import Rotation as R
from collections import deque
from torchvision import transforms


def get_image_augmentation(
    color_jitter = None,
    gray_scale: bool = True,
    gau_blur: bool = False
):
    """Create a composition of image augmentations.

    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.1,
        "p": 0.9
    }

    # Handle color jitter
    if color_jitter is not None:
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                    hue=effective_jitter["hue"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if gray_scale:
        transform_list.append(transforms.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )
    
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list) if transform_list else None

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def spatial_gradient(input, mode='sobel', order=1, normalized=True):
    """
    Compute the first order image derivative in both x and y using a Sobel operator.

    Args:
        input: input image tensor with shape (B, C, H, W).
        mode: derivatives modality, can be: `sobel` or `diff`.
        order: the order of the derivatives.
        normalized: whether the output is normalized.

    Return:
        the derivatives of the input feature map. with shape (B, C, 2, H, W).
    """
    def get_spatial_gradient_kernel2d(mode, order, device, dtype):
        if mode == 'sobel':
            kernel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=dtype, device=device)
            kernel_y = torch.tensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]], dtype=dtype, device=device)
        else:
            raise NotImplementedError(f"Mode {mode} not supported")
        return torch.stack([kernel_x, kernel_y], dim=0)

    def normalize_kernel2d(kernel):
        norm = kernel.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
        return kernel / norm

    kernel = get_spatial_gradient_kernel2d(mode, order, input.device, input.dtype)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...]

    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels = 3 if order == 2 else 2
    padded_inp = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, mode='replicate')
    out = F.conv2d(padded_inp, tmp_kernel, groups=1, padding=0, stride=1)
    return out.reshape(b, c, out_channels, h, w)

def get_surface_normal(points):
    '''
    Parameters:
        - points: (B, H, W, 3)
    '''
    # get gradient_x and gradient_y using sobel filter in pytorch
    points = points.permute(0, 3, 1, 2)
    
    gradients = spatial_gradient(points)  # Bx3x2xHxW

    # compute normals
    a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW

    # Use torch.cross without in-place operations
    normals = torch.cross(a, b, dim=1)  # Bx3xHxW

    # Normalize the normals without in-place operations
    normals = F.normalize(normals, dim=1, p=2)  # Bx3xHxW

    # Permute back to (B, H, W, 3)
    normals = normals.permute(0, 2, 3, 1).contiguous()
    
    return normals


def pad_or_crop_tsdf(tsdf, min_bound, voxel_size, target_shape=(96, 96, 56), pad_value=1.0):
    output = np.full(target_shape, pad_value, dtype=tsdf.dtype)
    mask = np.zeros(target_shape, dtype=np.bool_)

    input_shape = tsdf.shape
    slices_in = []
    slices_out = []
    offset_voxels = [0, 0, 0]  # How much the min_bound shifts in voxels

    for i in range(3):
        in_size = input_shape[i]
        out_size = target_shape[i]

        if in_size >= out_size:
            start = (in_size - out_size) // 2
            slices_in.append(slice(start, start + out_size))
            slices_out.append(slice(0, out_size))
            offset_voxels[i] = start
        else:
            start = (out_size - in_size) // 2
            slices_in.append(slice(0, in_size))
            slices_out.append(slice(start, start + in_size))
            offset_voxels[i] = -start  # negative shift because we're padding

    output[slices_out[0], slices_out[1], slices_out[2]] = tsdf[slices_in[0], slices_in[1], slices_in[2]]
    mask[slices_out[0], slices_out[1], slices_out[2]] = True

    offset = np.array(offset_voxels) * voxel_size
    new_min_bound = min_bound + offset
    new_max_bound = new_min_bound + np.array(target_shape) * voxel_size

    return output, mask, new_min_bound, new_max_bound


class BaseStereoViewDataset (EasyDataset):
    """ Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(self, *,  # only keyword arguments
                 split=None,
                 num_frames=5,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=None,
                 aug_crop=False,
                 seed=None):
        self.num_frames = num_frames
        self.split = split
        self._set_features(resolution, num_frames)

        if transform:
            self.transform = get_image_augmentation()
        else:
            self.transform = ImgNorm
        

        self.aug_crop = aug_crop
        self.seed = seed

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = '['+';'.join(f'{w}x{h}' for (w, h), t in self._features)+']'
        return f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace('self.', '').replace('\n', '').replace('   ', '')

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()
    

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._features) == 1, f"Multiple aspect ratios are not supported, but got {len(self._features)} features: {self._features}"
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution, num_frames = self._features[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, num_frames, self._rng)

        pts_all = None

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            if 'orig_img' in view:
                view['orig_img'] = np.asarray(view['orig_img'])

                if view['orig_img'].max() > 1.0:
                    view['orig_img'] = view['orig_img'].astype(np.float32) / 255.0
                
                if view['orig_img'].min() < 0.0 or view['orig_img'].max() > 1.0:
                    print(f"[warning] orig_img not in [0, 1], got min {view['orig_img'].min()}, max {view['orig_img'].max()} from dataset {view.get('dataset', 'Unknown Dataset')}")
                    raise ValueError(f"[warning] orig_img not in [0, 1], got min {view['orig_img'].min()}, max {view['orig_img'].max()} from dataset {view.get('dataset', 'Unknown Dataset')}")

            

            if view['camera_intrinsics'].shape == (4, 4):
                view['camera_intrinsics'] = view['camera_intrinsics'][:3, :3]  # remove the last row and column
                print(f"Warning: camera_intrinsics for view {view_name(view)} has shape (4, 4), but should be (3, 3). Removing the last row and column.")


            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            # TODO: Depth threshold of 5 please note here
            # if 'scannet' in views[0]['dataset']:
            #     valid_mask[view['depthmap'] > 3] = False
            
            # if 'mapfree' in views[0]['dataset']:
            #     valid_mask[view['depthmap'] > 3] = False

            # if 'normal' not in view:
            #     view['normal'] = get_surface_normal(torch.from_numpy(pts3d[None,...])).squeeze(0)




            view['pts3d'] = pts3d
            # view['pts3d_inv'] =  np.nan_to_num(1/pts3d, nan=0)
            # nan to 0
            
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

            if 'tsdf' in views[0].keys():
                if pts_all is None:
                    pts_all = pts3d[valid_mask]
                else:
                    pts_all = np.concatenate((pts_all, pts3d[valid_mask]), axis=0)



        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')


            
        images_all = torch.stack([view['img'] for view in views], dim=0)
        pts3d_all = torch.stack([torch.from_numpy(view['pts3d']) for view in views], dim=0)
        depth_all = torch.stack([torch.from_numpy(view['depthmap']) for view in views], dim=0)
        valid_mask_all = torch.stack([torch.from_numpy(view['valid_mask']) for view in views], dim=0)
        intrinsics_all = torch.stack([torch.from_numpy(view['camera_intrinsics']) for view in views], dim=0)
        camera_pose_all = torch.stack([torch.from_numpy(view['camera_pose']) for view in views], dim=0)

        if 'extrinsic' in views[0]:
            extrinsics_all = torch.stack([torch.from_numpy(view['extrinsic']) for view in views], dim=0)
        else:
            extrinsics_all = closed_form_inverse_se3(camera_pose_all)

        depth_all_ori = torch.stack([torch.from_numpy(view['depthmap_ori']) for view in views], dim=0) if 'depthmap_ori' in views[0] else None


        views_all = {
            'pts3d': pts3d_all,  # (B, H, W, 3)
            'depthmap': depth_all,  # (B, H, W)
            'valid_mask': valid_mask_all,  # (B, H, W)
            'camera_intrinsics': intrinsics_all,  # (B, 3, 3)
            'camera_pose': camera_pose_all,  # (B, 4, 4)
            'extrinsics': extrinsics_all,  # (B, 4, 4)
            'images': images_all,  # (B, 3, H, W)
        }

        if depth_all_ori is not None:
            views_all['depthmap_ori'] = depth_all_ori
            
            
        return (views, views_all)

    def _set_features(self, resolutions, num_frames_pool):
        if not isinstance(resolutions, list):
            resolutions = [resolutions]
        if not isinstance(num_frames_pool, list):
            num_frames_pool = [num_frames_pool]
        
        all_combinations = list(itertools.product(resolutions, num_frames_pool))
        self._features = []
        for res, nf in all_combinations:
            if isinstance(res, int):
                width = height = res
            else:
                width, height = res
            assert isinstance(width, int) and isinstance(height, int) and width >= height
            assert isinstance(nf, int)
            self._features.append(((width, height), nf))
    
    def _set_resolutions(self, resolutions):
        ''' Set the resolution(s) of the dataset.
        Params:
            - resolutions: int or tuple or list of tuples
        '''
        assert resolutions is not None, 'undefined resolution'

        if not isinstance(resolutions, list):
            resolutions = [resolutions]
            
            
        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            assert width >= height
            self._resolutions.append((width, height))

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
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        
        ## Recale with max factor, so  one of width or height might be larger than target_resolution
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2


def is_good_type(key, v):
    """ returns (is_good, err_msg) 
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view['dataset'])
    label = sel(view['label'])
    instance = sel(view['instance'])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        # rectify portrait to landscape
        assert view['img'].shape == (3, height, width)
        # view['img'] = view['img'].swapaxes(1, 2)
        view['img'] = torch.rot90(view['img'], k=-1, dims=(1, 2))


        assert view['valid_mask'].shape == (height, width)
        # view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)
        if isinstance(view['valid_mask'], np.ndarray):
            view['valid_mask'] = np.rot90(view['valid_mask'], k=-1, axes=(0, 1)).copy()
        else:
            view['valid_mask'] = torch.rot90(view['valid_mask'], k=-1, dims=(0, 1))

        assert view['depthmap'].shape == (height, width)
        # view['depthmap'] = view['depthmap'].swapaxes(0, 1)
        if isinstance(view['depthmap'], np.ndarray):
            view['depthmap'] = np.rot90(view['depthmap'], k=-1, axes=(0, 1)).copy()
        else:
            view['depthmap'] = torch.rot90(view['depthmap'], k=-1, dims=(0, 1))  # rotate depthmap


        assert view['pts3d'].shape == (height, width, 3)
        # view['pts3d'] = view['pts3d'].swapaxes(0, 1)
        if isinstance(view['pts3d'], np.ndarray):
            view['pts3d'] = np.rot90(view['pts3d'], k=-1, axes=(0, 1)).copy()
        else:
            view['pts3d'] = torch.rot90(view['pts3d'], k=-1, dims=(0, 1))  # rotate per-pixel 3D points


        # transpose x and y pixels
        # view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]
        view['camera_intrinsics'] = np.array([
                                                [view['camera_intrinsics'][1, 1], 0,       view['camera_intrinsics'][1, 2]],
                                                [0,       view['camera_intrinsics'][0, 0], view['camera_intrinsics'][0, 2]],
                                                [0,       0,       1      ]
                                            ])
        

        # R_z = np.array([[ 0, 1, 0],
        #         [-1, 0, 0],
        #         [ 0, 0, 1]])  # Rotation around Z axis by -90°
        R_z_inv = np.array([[ 0, -1, 0],
                            [ 1,  0, 0],
                            [ 0,  0, 1]])
        
        view['camera_pose'][:3, :3] = view['camera_pose'][:3, :3] @ R_z_inv.T

        if 'extrinsic' in view:
            # This matrix should be a 4x4 world-to-camera matrix.
            extrinsic = view['extrinsic']
            
            # Define the +90 degree rotation matrix around the Z-axis in a 4x4 form.
            # This matrix will transform points from the old camera frame to the new one.
            R_fix_w2c = np.array([
                [0, -1, 0, 0],
                [1,  0, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1]
            ])
            
            # Pre-multiply the W2C extrinsic matrix by the fix-up rotation.
            # This correctly rotates both the rotation and translation components.
            view['extrinsic'] = (R_fix_w2c @ extrinsic).astype(np.float32)

            # assert np.allclose(view['extrinsic'], np.linalg.inv(view['camera_pose'])), f"Extrinsic: {view['extrinsic']}, Camera Pose inv: {np.linalg.inv(view['camera_pose'])} are not consistent after transposition for {view_name(view)}"
        # view['camera_pose'][:3, 3] = R_z @ view['camera_pose'][:3, 3]
        # view['camera_pose'][:3, :3] = view['camera_pose'][:3, :3] @ R_z_inv

        # # The translation vector is also rotated relative to the new frame.
        # # We also need to apply the rotation to the translation component.
        # view['camera_pose'][:3, 3] = view['camera_pose'][:3, 3] @ R_z_inv



        # print(f"Camera pose before transposition: {view['camera_pose']}")
        # # view['camera_pose'][:3, :3] = view['camera_pose'][:3, :3] @ R.from_euler('z', 90, degrees=True).as_matrix()
        # view['camera_pose'][:3, :3] = R.from_euler('z', -90, degrees=True).as_matrix() @ view['camera_pose'][:3, :3]
        # view['camera_pose'][:3, 3] = R.from_euler('z', -90, degrees=True).as_matrix() @ view['camera_pose'][:3, 3]
        # print(f"Camera pose after transposition: {view['camera_pose']}")

        # print(view['camera_pose'].shape)
        # raise NotImplementedError(f"Camera pose transposition is not implemented for {view_name(view)}")
