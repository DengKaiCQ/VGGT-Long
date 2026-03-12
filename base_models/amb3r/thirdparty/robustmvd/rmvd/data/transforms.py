# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/transforms.py

import torch
import numpy as np
from skimage.transform import resize
from PIL import Image
import torchvision.transforms
import cv2

from rmvd.utils import trans_from_transform, rot_from_transform, transform_from_rot_trans, compute_depth_range, to_numpy


class Bernoulli:
    def __init__(self, prob):
        self.prob = prob

    def sample(self, size=1):
        return np.random.binomial(n=1, p=self.prob, size=size)


class UniformBernoulli:
    def __init__(self, mean, spread, prob=1., exp=False):
        self.mean = mean
        self.spread = spread
        self.prob = prob
        if exp:
            self.sample = self.sample_exp
        else:
            self.sample = self.sample_noexp

    def sample_noexp(self, size=1):
        gate = Bernoulli(self.prob).sample(size)
        return gate * np.random.uniform(low=self.mean - self.spread, high=self.mean + self.spread, size=size)

    def sample_exp(self, size):
        gate = Bernoulli(self.prob).sample(size=1)
        return gate * np.exp(np.random.uniform(low=self.mean - self.spread, high=self.mean + self.spread, size=size))


class ResizeInputs:
    """Resize sample inputs to given size (height, width).

    Args:
        size (tuple): tuple with input size (height, width).
    """
    def __init__(
            self,
            size,
            interpolation_order=1,
    ):
        self.__height = size[0]
        self.__width = size[1]
        self.__interpolation_order = interpolation_order

    def __call__(self, sample):
        image = sample["images"][0]
        orig_ht, orig_wd = image.shape[-2:]
        
        wd, ht = self.__width, self.__height
        # resize images
        if "images" in sample:
            images = sample["images"]
            images = [resize(image, list(image.shape[:-2]) + [ht, wd], order=self.__interpolation_order) for image in images]
            sample["images"] = images
        
        # resize intrinsics:
        if "intrinsics" in sample:
            scale_arr = np.array([[wd / orig_wd]*3, [ht / orig_ht]*3, [1.]*3], dtype=np.float32)  # 3, 3
            sample["intrinsics"] = [intrinsic * scale_arr for intrinsic in sample["intrinsics"]]
            
        return sample
    

class ResizeTargets:
    def __init__(
            self,
            size,
            interpolation_order=0,
    ):
        self.__height = size[0]
        self.__width = size[1]
        self.__interpolation_order = interpolation_order
        
    def __call__(self, sample):
        wd, ht = self.__width, self.__height
        # resize depth:
        if "depth" in sample:
            depth = sample["depth"]
            depth = resize(depth, list(depth.shape[:-2]) + [ht, wd], order=self.__interpolation_order)
            sample["depth"] = depth
        
        # resize invdepth:
        if "invdepth" in sample:
            invdepth = sample["invdepth"]
            invdepth = resize(invdepth, list(invdepth.shape[:-2]) + [ht, wd], order=self.__interpolation_order)
            sample["invdepth"] = invdepth
        
        if "depth_range" in sample:
            depth_range = compute_depth_range(depth=sample.get("depth", None), invdepth=sample.get("invdepth", None))
            sample["depth_range"] = depth_range
        
        return sample
    
    
class SpatialAugmentation:
    def __init__(
            self,
            size,
            p,
            stretch_p=0.,
            max_stretch=0.2,
    ):
        self.__height = size[0]
        self.__width = size[1]
        self.__p = p
        self.__stretch_p = stretch_p
        self.__max_stretch = max_stretch
        
    def __call__(self, sample):
        images = sample["images"]  # 3, H, W, float32
        cht, cwd = self.__height, self.__width
        ht, wd = images[0].shape[-2:]
        
        if np.random.rand() < self.__p:

            min_scale = np.maximum((cht + 8) / float(ht), (cwd + 8) / float(wd))

            scale = UniformBernoulli(mean=0.2, spread=0.4, exp=True).sample(1)[0] * \
                    UniformBernoulli(mean=0., spread=0.3, exp=True).sample(1)[0]
            for i_ in range(5):
                if scale < 1.2 and np.random.rand() < 0.9:  # mimic the validity check in the old tf code
                    scale = UniformBernoulli(mean=0.2, spread=0.4, exp=True).sample(1)[0] * \
                            UniformBernoulli(mean=0., spread=0.3, exp=True).sample(1)[0]
                else:
                    break

            scale_x = scale
            scale_y = scale

            if np.random.rand() < self.__stretch_p:
                scale_x *= 2 ** np.random.uniform(-self.__max_stretch, self.__max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.__max_stretch, self.__max_stretch)

            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

            sht = None
            swd = None
            
            # resize images
            if "images" in sample:
                images = sample["images"]
                images = [np.transpose(image, [1, 2, 0]) for image in images]  # H, W, 3
                images = [cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR) for image in images]
                images = [np.transpose(image, [2, 0, 1]) for image in images]  # 3, H, W
                sht, swd = images[0].shape[-2:]
                sample["images"] = images
        
            # resize intrinsics:
            if "intrinsics" in sample:
                scale_arr = np.array([[swd / wd]*3, [sht / ht]*3, [1.]*3], dtype=np.float32)  # 3, 3
                sample["intrinsics"] = [intrinsic * scale_arr for intrinsic in sample["intrinsics"]]
                
            # resize depth:
            if "depth" in sample:
                depth = sample["depth"][0]
                depth = cv2.resize(depth, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
                depth = depth[None, ...]
                if sht is None:
                    sht, swd = depth.shape[-2:]
                else:
                    assert sht == depth.shape[-2] and swd == depth.shape[-1]
                sample["depth"] = depth
            
            # resize invdepth:
            if "invdepth" in sample:
                invdepth = sample["invdepth"][0]
                invdepth = cv2.resize(invdepth, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
                invdepth = invdepth[None, ...]
                if sht is None:
                    sht, swd = invdepth.shape[-2:]
                else:
                    assert sht == invdepth.shape[-2] and swd == invdepth.shape[-1]
                sample["invdepth"] = invdepth
                
            y0 = np.random.randint(0, sht - cht)
            x0 = np.random.randint(0, swd - cwd)
            
            # crop images
            if "images" in sample:
                images = sample["images"]
                images = [image[:, y0:y0+cht, x0:x0+cwd] for image in images]
                sample["images"] = images
                
            # crop intrinsics:
            if "intrinsics" in sample:
                intrinsics = sample["intrinsics"]
                shift_arr = np.array([[0, 0, -x0], [0, 0, -y0], [0.] * 3], dtype=np.float32)  # 3, 3
                intrinsics = [intrinsic + shift_arr for intrinsic in intrinsics]
                sample["intrinsics"] = intrinsics
                
            if "depth" in sample:
                depth = sample["depth"]
                depth = depth[:, y0:y0+cht, x0:x0+cwd]
                sample["depth"] = depth
                
            if "invdepth" in sample:
                invdepth = sample["invdepth"]
                invdepth = invdepth[:, y0:y0+cht, x0:x0+cwd]
                sample["invdepth"] = invdepth
                
            if "depth_range" in sample:
                depth_range = compute_depth_range(depth=sample.get("depth", None), invdepth=sample.get("invdepth", None))
                sample["depth_range"] = depth_range
                
        return sample
    

class ColorJitter:
    """Apply ColorJitter from torchvision to all images in a sample."""
    def __init__(self, **kwargs):
        self.color_aug = torchvision.transforms.ColorJitter(**kwargs)
        
    def __call__(self, sample):
        images = sample["images"]
        images = [np.transpose(image, [1, 2, 0]) for image in images]  # H, W, 3
        num_images = len(images)
        image_stack = np.concatenate(images, axis=0)  # H*num_images, W, 3
        image_stack = image_stack.astype(np.uint8)
        image_stack = np.array(self.color_aug(Image.fromarray(image_stack)), dtype=np.float32)  # H*num_images, W, 3
        images = np.split(image_stack, num_images, axis=0)
        images = [np.transpose(image, [2, 0, 1]) for image in images]  # 3, H, W
        sample["images"] = images
        return sample


class NormalizeImagesToMinMax(object):
    """Normalize images to range [min_val, max_val]."""
    def __init__(self, min_val, max_val):
        self.__min_val = min_val
        self.__max_val = max_val

    def __call__(self, sample):
        images = sample["images"]  # 3, H, W, float32, range [0, 255]
        images = [image / 255.0 for image in images]  # 3, H, W, float32, range [0, 1]
        images = [image * (self.__max_val - self.__min_val) + self.__min_val for image in images]  # 3, H, W, float32, range [min_val, max_val]
        sample["images"] = images
        return sample
    

class Eraser:
    def __init__(self, bounds, p):
        self.__bounds = bounds
        self.__p = p
        
    def __call__(self, sample):
        images = sample["images"]  # 3, H, W, float32
        keyview_idx = sample["keyview_idx"]
        src_indices = [i for i in range(len(images)) if i != keyview_idx]
        
        for src_idx in src_indices:
            image_src = images[src_idx]
            if np.random.rand() < self.__p:
                mean_color = np.mean(image_src.reshape(3, -1), axis=-1)
                ht, wd = image_src.shape[-2:]
                for _ in range(np.random.randint(1, 3)):
                    dx = np.random.randint(self.__bounds[0], self.__bounds[1])
                    dy = np.random.randint(self.__bounds[0], self.__bounds[1])
                    x0 = np.random.randint(0, wd)
                    y0 = np.random.randint(0, ht)

                    min_x = max(0, x0 - dx//2)
                    max_x = min(wd-1, x0 + dx//2)

                    min_y = max(0, y0 - dy//2)
                    max_y = min(ht-1, y0 + dy//2)

                    image_src = image_src.transpose(1, 2, 0)
                    image_src[min_y:max_y, min_x:max_x, :] = mean_color
                    image_src = image_src.transpose(2, 0, 1)
                images[src_idx] = image_src
                
        sample["images"] = images
        return sample
    

class Scale3DFixed:
    def __init__(self, scale, p):
        self.__scale = scale
        self.__p = p
        
    def __call__(self, sample):
        if np.random.rand() < self.__p:
            poses = sample["poses"]
            depth = sample["depth"]
            invdepth = sample["invdepth"]
            depth_range = sample["depth_range"]
            
            scale_factor = self.__scale
            for idx, pose in enumerate(poses):  # pose is 4, 4, float32
                trans = trans_from_transform(pose) * scale_factor
                poses[idx] = transform_from_rot_trans(rot_from_transform(pose), trans)
            depth = depth * scale_factor
            invdepth = invdepth / scale_factor
            depth_range = (depth_range[0] * scale_factor, depth_range[1] * scale_factor)
            
            sample["poses"] = poses
            sample["depth"] = depth
            sample["invdepth"] = invdepth
            sample["depth_range"] = depth_range
            
        return sample


class MaskDepth:
    def __init__(self, min_depth, max_depth):
        self.__min_depth = min_depth
        self.__max_depth = max_depth
        
    def __call__(self, sample):
        depth = sample["depth"]
        invdepth = sample["invdepth"]
        mask = ((depth >= self.__min_depth) & (depth <= self.__max_depth)).astype(np.float32)
        depth = depth * mask
        invdepth = invdepth * mask
        depth_range = compute_depth_range(depth=depth)
            
        sample["depth"] = depth
        sample["invdepth"] = invdepth
        sample["depth_range"] = depth_range
        
        return sample


class NormalizeIntrinsics:
    def __call__(self, sample):
        image = sample["images"][0]
        ht, wd = image.shape[-2:]
        
        if "intrinsics" in sample:
            scale_arr = np.array([[1/wd]*3, [1/ht]*3, [1.]*3], dtype=np.float32)  # 3, 3
            sample["intrinsics"] = [intrinsic * scale_arr for intrinsic in sample["intrinsics"]]
            
        return sample
