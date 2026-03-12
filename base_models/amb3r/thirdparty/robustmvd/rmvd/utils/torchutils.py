# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/utils/torchutils.py

import torch


def transform_from_rot_trans(R, t, dtype=torch.float32):
    """Transformation matrix from rotation matrix and translation vector."""

    if R.ndim == 4:
        R = R.squeeze(1)
    if t.ndim == 4:
        t = t.squeeze(1)

    R = R.to(dtype)
    t = t.to(dtype)

    if R.ndim == 2:
        if t.ndim == 1:
            t = t.reshape(3, 1)
        elif t.ndim != 2:
            raise ValueError("When the rotation has no batch dimension, the translation must have 1 or 2 dimensions, but has {} dimensions.".format(t.ndim))

        R = R.reshape(3, 3)
        T = torch.vstack((torch.hstack([R, t]), torch.tensor([0, 0, 0, 1], device=R.device))).to(dtype)

    elif R.ndim == 3:
        N = R.shape[0]

        if t.ndim == 2:
            t = t.reshape(-1, 3, 1)
        elif t.ndim != 3:
            raise ValueError("When the rotation has a batch dimension, the translation must have 2 or 3 dimensions, but has {} dimensions.".format(t.ndim))

        R = R.reshape(-1, 3, 3)
        T = torch.cat((R, t), 2).to(dtype)
        last_row = torch.cat([torch.tensor([0, 0, 0, 1], dtype=dtype, device=T.device).unsqueeze(0).unsqueeze(0)]*N)
        T = torch.cat((T, last_row), 1)

    else:
        raise ValueError("Rotation must have 2 or 3 dimensions (after squeezing), but has {} dimensions.".format(R.ndim))

    return T


def trans_from_transform(T):
    if T.ndim == 4:
        T = T.squeeze(1)

    if T.ndim == 2:
        t = T[0:3, 3]
    elif T.ndim == 3:
        t = T[:, 0:3, 3]
    else:
        raise ValueError("Transform must have 2 or 3 dimensions (after squeezing), but has {} dimensions.".format(T.ndim))

    return t


def rot_from_transform(T):
    if T.ndim == 4:
        T = T.squeeze(1)

    if T.ndim == 2:
        R = T[0:3, 0:3]
    elif T.ndim == 3:
        R = T[:, 0:3, 0:3]
    else:
        raise ValueError("Transform must have 2 or 3 dimensions, but has {} dimensions.".format(T.ndim))

    return R


def invert_transform(T):
    R = rot_from_transform(T)
    t = trans_from_transform(T).unsqueeze(-1)
    R_inv = torch.transpose(R, -1, -2)
    t_inv = torch.matmul(-R_inv, t)
    return transform_from_rot_trans(R_inv, t_inv, dtype=T.dtype)


def compute_depth_range(depth=None, invdepth=None, default_min_depth=0.1, default_max_depth=100.):
    assert depth is not None or invdepth is not None, "Either depth or invdepth must be provided."
    
    if depth is not None:
        mask = depth > 0
        
        depths = torch.split(depth, [1]*depth.shape[0])
        masks = torch.split(mask, [1]*mask.shape[0])
        min_depths = []
        max_depths = []
        for depth_sample, mask_sample in zip(depths, masks):
            if mask_sample.any():
                min_depth = torch.min(depth_sample[mask_sample])
                max_depth = torch.max(depth_sample[mask_sample])
            else:
                min_depth = torch.tensor(default_min_depth, dtype=depth_sample.dtype, device=depth_sample.device)
                max_depth = torch.tensor(default_max_depth, dtype=depth_sample.dtype, device=depth_sample.device)
            min_depths.append(min_depth)
            max_depths.append(max_depth)
                
        min_depth = torch.stack(min_depths)
        max_depth = torch.stack(max_depths)
                
        return [min_depth, max_depth]
        
    if invdepth is not None:
        mask = invdepth > 0
        
        invdepths = torch.split(invdepth, [1]*invdepth.shape[0])
        masks = torch.split(mask, [1]*mask.shape[0])
        min_depths = []
        max_depths = []
        for invdepth_sample, mask_sample in zip(invdepths, masks):
            if mask_sample.any():
                min_invdepth = torch.min(invdepth_sample[mask_sample])
                max_invdepth = torch.max(invdepth_sample[mask_sample])
            else:
                min_invdepth = torch.tensor(1./default_max_depth, dtype=invdepth_sample.dtype, device=invdepth_sample.device)
                max_invdepth = torch.tensor(1./default_min_depth, dtype=invdepth_sample.dtype, device=invdepth_sample.device)
            min_depth = 1./max_invdepth
            max_depth = 1./min_invdepth
            min_depths.append(min_depth)
            max_depths.append(max_depth)
                
        min_depth = torch.stack(min_depths)
        max_depth = torch.stack(max_depths)
                
        return [min_depth, max_depth]
