# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/data/batch_transforms.py

import torch
import numpy as np

from rmvd.utils import to_numpy
from rmvd.utils.torchutils import trans_from_transform, rot_from_transform, transform_from_rot_trans, compute_depth_range
    
    
class Histogram:
    def __init__(self, range, num_bins, logarithmic_bin_sizes=False, exclude_inf=False):
        self.exclude_inf = exclude_inf

        if not logarithmic_bin_sizes:
            self.bins = list(np.histogram_bin_edges(np.random.random(10), bins=num_bins, range=(range[0], range[1])))
        else:
            self.bins = list(np.logspace(np.log10(range[0]),np.log10(range[1]), num_bins+1))
        self.bins = [-np.inf] + self.bins + [np.inf] if not exclude_inf else self.bins
        
        self.counts = np.zeros(len(self.bins) - 1, dtype=np.int64)

    def add_values(self, tensor, mask=None):
        tensor = to_numpy(tensor)
        mask = to_numpy(mask) if mask is not None else mask
        masked_tensor = tensor[mask.astype(np.bool)] if mask is not None else tensor
        self.counts += np.histogram(masked_tensor, bins=self.bins)[0]

    @property
    def bin_centers(self):
        bins = self.bins[1:-1] if not self.exclude_inf else self.bins
        bin_centers = [bins[i]+(bins[i+1]-bins[i])/2. for i in range(len(bins)-1)]
        bin_centers = [-np.inf] + bin_centers + [np.inf] if not self.exclude_inf else bin_centers
        return bin_centers

    @property
    def bin_ranges(self):
        return [(self.bins[i], self.bins[i+1]) for i in range(len(self.bins)-1)]


class Scale3DEqualizedBatch:
    def __init__(self, p, min_depth, max_depth):
        self.__p = p
        self.__counter = 0
        self.depth_histogram = Histogram(range=(min_depth, max_depth), num_bins=100, logarithmic_bin_sizes=True)
        
    def __call__(self, sample):
        poses = sample["poses"]
        depth = sample["depth"]
        invdepth = sample["invdepth"]
        depth_range = sample["depth_range"]
        depth_mask = depth > 0
            
        if np.random.rand() < self.__p:
            
            if self.__counter > 10:
                bin_idx = self.depth_histogram.counts[1:-1].argmin()
                bin_min, bin_max = self.depth_histogram.bin_ranges[bin_idx]
                if not np.isfinite(bin_min):
                    bin_val = bin_max
                elif not np.isfinite(bin_max):
                    bin_val = bin_min
                else:
                    bin_val = np.random.uniform(bin_min, bin_max)

                depths = torch.split(depth, [1]*depth.shape[0])  # tuple of len N with depths of shape 1, 1, H, W
                depth_masks = torch.split(depth_mask, [1]*depth_mask.shape[0])
                scale_factors = []

                for depth_sample, depth_mask_sample in zip(depths, depth_masks):
                    masked_depth_sample = depth_sample[depth_mask_sample.bool()]
                    if len(masked_depth_sample) > 0:
                        scale_factor = bin_val / torch.median(masked_depth_sample)
                        scale_factor = torch.nan_to_num(scale_factor, nan=1., posinf=1., neginf=1.)
                    else:
                        scale_factor = torch.tensor(1, dtype=depth_sample.dtype, device=depth_sample.device)
                    scale_factors.append(scale_factor)

                scale_factor = torch.stack(scale_factors)
                scale_factor.unsqueeze_(-1)  # N, 1
        
                for idx, pose in enumerate(poses):  # pose is N, 4, 4
                    trans = trans_from_transform(pose)  # N, 3
                    trans = trans * scale_factor
                    rot = rot_from_transform(pose)
                    poses[idx] = transform_from_rot_trans(rot, trans)
                depth = depth * scale_factor[..., None, None]
                invdepth = invdepth / scale_factor[..., None, None]
                depth_range = compute_depth_range(depth=depth)
                
                sample["poses"] = poses
                sample["depth"] = depth
                sample["invdepth"] = invdepth
                sample["depth_range"] = depth_range
            
        self.__counter += 1
        self.depth_histogram.add_values(depth, depth_mask)
            
        return sample


class MaskDepth:
    def __init__(self, min_depth, max_depth):
        self.__min_depth = min_depth
        self.__max_depth = max_depth
        
    def __call__(self, sample):
        depth = sample["depth"]
        invdepth = sample["invdepth"]
        mask = ((depth >= self.__min_depth) & (depth <= self.__max_depth)).to(torch.float32)
        depth = depth * mask
        invdepth = invdepth * mask
        depth_range = compute_depth_range(depth=depth)
            
        sample["depth"] = depth
        sample["invdepth"] = invdepth
        sample["depth_range"] = depth_range
        
        return sample
