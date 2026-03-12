# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/tools/pts_align.py

import torch
from moge.moge.train.losses import scale_invariant_alignment
from amb3r.tools.pose_align import average_transforms_with_weights

def robust_scale_invariant_alignment(pts_src, pts_tgt, mask=None, trunc=None, scale=None, robust=True):
    '''
    A robust version of scale-invariant alignment that considers multiple scale candidates and 
    selects the best one based on relative z-error. 
    '''
    
    pts_src_scaled1, scale1 = scale_invariant_alignment(
                                pts_src.contiguous().view(1, -1, 3),
                                pts_tgt.contiguous().view(1, -1, 3),
                                mask.view(1, -1) if mask is not None else None,
                                trunc=trunc)
    pts_src_scaled1 = pts_src_scaled1.view_as(pts_src)

    if not robust:
        return pts_src_scaled1, scale1

    # Collect all scale candidates: (scaled_pts, scale_value)
    scale2 = (pts_tgt[mask].norm(dim=-1) / pts_src[mask].norm(dim=-1)).median()


    candidates = [
        (pts_src_scaled1, scale1),
        (pts_src * scale2, scale2),
    ]
    if scale is not None:
        candidates.append((pts_src * scale, scale))

    # Pick the candidate with the smallest mean relative z-error
    tgt_dist = pts_tgt.norm(dim=-1)
    best_pts, best_scale = candidates[0]
    best_err = float('inf')

    for pts_candidate, scale_candidate in candidates:
        rel_error = (pts_candidate[..., -1:] - pts_tgt[..., -1:]).norm(dim=-1) / tgt_dist
        err = rel_error[mask].mean()
        if err < best_err:
            best_err = err
            best_pts, best_scale = pts_candidate, scale_candidate

    return best_pts, best_scale


def transform_pts_global_to_local(pts, pose):
    '''
    Transform points from global coordinates to local coordinates based on a reference pose.
    
    Args:
        pts (torch.Tensor): The points in global coordinates. Shape: (T, H, W, 3)
        pose (torch.Tensor): The reference pose. Shape: (T, 4, 4)
    
    Returns:
        torch.Tensor: The points transformed to local coordinates. Shape: (T, H, W, 3)
    '''
    ref_pose = pose[0]
    
    global_to_local_transform = torch.inverse(ref_pose)
    T_kf, H, W, _ = pts.shape
    pts_flat = pts.reshape(-1, 3)
    pts_transformed_flat = (global_to_local_transform[:3, :3] @ pts_flat.T).T + global_to_local_transform[:3, 3]
    pts = pts_transformed_flat.reshape(T_kf, H, W, 3)
    
    return pts



def transform_pts_and_poses_global_to_local(pts, pose):
    '''
    Transform points and poses from global coordinates to local coordinates based on a reference pose.
    
    Args:
        pts (torch.Tensor): The points in global coordinates. Shape: (T, H, W, 3)
        pose (torch.Tensor): The poses. Shape: (T, 4, 4)
        
    Returns:
        pts_local (torch.Tensor): The points transformed to local coordinates. Shape: (T, H, W, 3)
        poses_local (torch.Tensor): The poses transformed to local coordinates. Shape: (T, 4, 4)
    '''
    # pts: (T, H, W, 3)
    # pose: (T, 4, 4)
    ref_pose = pose[0]
    
    global_to_local_transform = torch.inverse(ref_pose)
    T_kf, H, W, _ = pts.shape
    pts_flat = pts.reshape(-1, 3)
    pts_transformed_flat = (global_to_local_transform[:3, :3] @ pts_flat.T).T + global_to_local_transform[:3, 3]
    pts = pts_transformed_flat.reshape(T_kf, H, W, 3)

    poses_local = global_to_local_transform @ pose
    
    return pts, poses_local


def coordinate_alignment(pts_local, c2w_local, conf_local, 
                         pts_global, c2w_global, conf_global, 
                         num_kf, transform=True,
                         trunc=None,
                         scale=None):
    ''' Align local coordinates to global coordinates using a robust scale-invariant alignment 
    followed by a weighted average pose alignment.
    
    Args:
        - pts_local: (T, H, W, 3) tensor of local points
        - c2w_local: (T, 4, 4) tensor of local camera-to-world poses
        - conf_local: (T, H, W) tensor of confidence scores for local points
        - pts_global: (T, H, W, 3) tensor of global points
        - c2w_global: (T, 4, 4) tensor of global camera-to-world poses
        - conf_global: (T, H, W) tensor of confidence scores for global points
        - num_kf: number of keyframes to use for alignment
        - transform: whether to perform the initial global-to-local transformation (default: True)
        - trunc: truncation value for scale-invariant alignment (optional)
        - scale: initial scale guess for scale-invariant alignment (optional)
    
    Returns:
        - pts_global_from_local: (T, H, W, 3) tensor of points transformed from local to global coordinates
        - c2w_global_from_local: (T, 4, 4) tensor of poses transformed from local to global coordinates
    
    '''
    # ref_kf_id = self.cur_kf_idx[0]
    # Step 1: Transform global points to local coordinates
    if transform:
        pts_local_from_global = transform_pts_global_to_local(pts_global, c2w_global)
        pts_kf_local_from_global = pts_local_from_global[:num_kf]
    else:
        pts_kf_local_from_global = pts_global[:num_kf]

    pts_kf_local = pts_local[:num_kf]
    c2w_kf_local = c2w_local[:num_kf]
    conf_kf_local = conf_local[:num_kf]

    # Step 2: Scale estimation
    T_kf, H, W, _ = pts_kf_local.shape

    valid_scale_mask = (conf_kf_local.contiguous() >= torch.quantile(conf_kf_local.contiguous(), 0.5)).view(1, T_kf * H, W)
    _, kf_scale = robust_scale_invariant_alignment(
                                            pts_kf_local.contiguous().view(1, T_kf * H, W, 3),
                                            pts_kf_local_from_global.contiguous().view(1, T_kf * H, W, 3),
                                            valid_scale_mask,
                                            scale=scale,
                                            trunc=trunc)

    pts_local_scaled = pts_local * kf_scale
    
    # Only scale the translation part of the pose
    c2w_local_scaled = c2w_local.clone()
    c2w_local_scaled[..., :3, 3] *= kf_scale
    c2w_kf_local_scaled = c2w_local_scaled[:num_kf]


    # Step 3: Weighted average alignment

    c2w_kf_global = c2w_global[:num_kf]
    conf_kf_global = conf_global[:num_kf]
    
    if transform:
        pose_relative = c2w_kf_global @ torch.inverse(c2w_kf_local_scaled)

        conf_bi = conf_kf_global * conf_kf_local
        weights = conf_bi.mean(dim=(1, 2))
        pose_align = average_transforms_with_weights(pose_relative, weights)

        R_align = pose_align[:3, :3]
        t_align = pose_align[:3, 3]

        c2w_global_from_local = pose_align @ c2w_local_scaled

        T, H, W, _ = pts_local_scaled.shape
        pts_local_scaled_flat = pts_local_scaled.reshape(-1, 3)
        pts_global_from_local_flat = (R_align @ pts_local_scaled_flat.T).T + t_align
        pts_global_from_local = pts_global_from_local_flat.reshape(T, H, W, 3)
    
    else:
        pts_global_from_local = pts_local_scaled
        c2w_global_from_local = c2w_local_scaled
    

    return pts_global_from_local, c2w_global_from_local

