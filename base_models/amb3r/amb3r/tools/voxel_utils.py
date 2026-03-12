# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/tools/voxel_utils.py

import torch

def get_vox_indices(points, batch_ids, voxel_size, bounding_boxes, shift=False, cat_batch_ids=True):
    '''Compute voxel indices for a batch of points.
    
    Args:
        points (torch.Tensor): A batch of points with shape (N, 3).
        batch_ids (torch.Tensor): A tensor of batch IDs for each point with shape (N,).
        voxel_size (torch.Tensor | float): The size of each voxel.
        bounding_boxes (torch.Tensor): The bounding boxes for each batch with shape (B, 2, 3).
        shift (bool): Whether to shift the voxel indices by half a voxel.
        cat_batch_ids (bool): Whether to concatenate the batch IDs to the voxel indices.

    Returns:
        voxel_indices (torch.Tensor): The computed voxel indices with shape (N, 4) if cat_batch_ids is True, else (N, 3).
    '''
    
    bb_mins = bounding_boxes[batch_ids][:, 0]  # select min box per point
    voxel_size_b = voxel_size[batch_ids] if isinstance(voxel_size, torch.Tensor) else voxel_size
    if shift:
        bb_mins = bb_mins - 0.5 * voxel_size
    
    voxel_indices = torch.floor((points - bb_mins) / voxel_size_b).long()

    if cat_batch_ids:
        voxel_indices = torch.cat([batch_ids.unsqueeze(1), voxel_indices], dim=1)
    
    return voxel_indices

def get_vox_centers_from_indices(voxel_indices, batch_ids, voxel_size, bounding_boxes, shift=False, cat_batch_ids=True):
    '''Compute voxel centers from voxel indices.
    
    Args:
        voxel_indices (torch.Tensor): The voxel indices with shape (N, 4)
        batch_ids (torch.Tensor): A tensor of batch IDs for each voxel with shape (N,).
        voxel_size (torch.Tensor | float): The size of each voxel.
        bounding_boxes (torch.Tensor): The bounding boxes for each batch with shape (B, 2, 3).
        shift (bool): Whether to shift the voxel centers by half a voxel.
        cat_batch_ids (bool): Whether the voxel indices include batch IDs.

    Returns:
        voxel_centers (torch.Tensor): The computed voxel centers with shape (N, 3).
    '''

    bb_mins = bounding_boxes[batch_ids][:, 0]
    if shift and cat_batch_ids:
        voxel_centers = voxel_indices[:, 1:] * voxel_size + bb_mins
    
    elif (not shift) and cat_batch_ids:
        voxel_centers = voxel_indices[:, 1:] * voxel_size + bb_mins + 0.5 * voxel_size

    else:
        raise NotImplementedError("Only shift and cat_batch_ids are supported")

    return voxel_centers

