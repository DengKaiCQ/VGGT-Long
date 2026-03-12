# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/tools/pose_align.py

import torch
import numpy as np
from scipy.spatial.transform import Rotation

def umeyama_alignment(pts_pred, pts_gt, valid_mask=None, eps=1e-8, scale=None):
    """
    Performs Kabsch-Umeyama alignment with optional scale for a single point cloud.

    Args:
        pts_pred:   (N, 3) tensor of predicted points.
        pts_gt:     (N, 3) tensor of ground-truth points.
        valid_mask: (N,) boolean tensor indicating valid points.
        eps:        Small constant to avoid division by zero.

    Returns:
        aligned_pred: (N, 3) tensor of aligned points.
        info:         Dictionary with 'R' (3, 3), 't' (3,), 'scale' (scalar).
    """
    # Ensure correct shapes
    pts_pred = pts_pred.reshape(-1, 3)
    pts_gt = pts_gt.reshape(-1, 3)

    if valid_mask is None:
        valid_mask = torch.ones(pts_pred.shape[0], dtype=torch.bool, device=pts_pred.device)
    else:
        valid_mask = valid_mask.reshape(-1).to(torch.bool)

    # Check for valid points
    if valid_mask.sum() < 3:
        raise ValueError(f"Not enough valid points for alignment! Got {valid_mask.sum().item()} points.")
    if not torch.isfinite(pts_pred).all() or not torch.isfinite(pts_gt).all():
        raise ValueError("Non-finite values detected in input points!")

    # Select valid points
    X = pts_pred[valid_mask]
    Y = pts_gt[valid_mask]

    # Mean center
    mu_X = X.mean(dim=0)
    mu_Y = Y.mean(dim=0)
    Xc = X - mu_X
    Yc = Y - mu_Y

    # Covariance
    cov = Xc.T @ Yc
    if not torch.isfinite(cov).all():
        raise ValueError("Covariance matrix has non-finite values!")

    # SVD
    U, _, V = torch.svd(cov)   # like their code

    # Reflection correction
    Z = torch.eye(3, device=U.device)
    Z[2, 2] = torch.sign(torch.linalg.det(U @ V.T))
    R = V @ Z @ U.T

    # Scale (Umeyama)
    if scale is None:
        var_X = (Xc ** 2).sum()
        scale = (torch.trace(R @ cov) / var_X).clamp_min(eps)


    # Translation
    t = mu_Y - scale * (mu_X @ R.T)

    # Apply alignment
    aligned_pred = scale * (pts_pred @ R.T) + t

    info = {
        'R': R,         # (3, 3)
        't': t,         # (3,)
        'scale': scale  # scalar
    }
    return aligned_pred, info


def average_transforms_with_weights(transforms, weights):
    """
    Calculates a weighted average of multiple 4x4 transformation matrices
    using a pre-computed vector of weights.
    """
    # 1. Normalize weights to sum to 1
    if torch.sum(weights) > 0:
        weights = weights / torch.sum(weights)
    else:
        weights = torch.ones_like(weights) / len(weights)

    # 2. Weighted average of the translation component
    translations = transforms[:, :3, 3]
    avg_translation = torch.sum(weights.view(-1, 1) * translations, dim=0)

    # 3. Weighted average of the rotation component (using quaternions)
    rot_matrices_np = transforms[:, :3, :3].cpu().numpy()
    quats_np = Rotation.from_matrix(rot_matrices_np).as_quat()

    # Align quaternions to handle the double-cover problem
    for i in range(1, len(quats_np)):
        if np.dot(quats_np[0], quats_np[i]) < 0:
            quats_np[i] *= -1

    avg_quat_np = np.sum(weights.cpu().numpy().reshape(-1, 1) * quats_np, axis=0)
    avg_quat_np /= np.linalg.norm(avg_quat_np)

    avg_rot_matrix = torch.from_numpy(
        Rotation.from_quat(avg_quat_np).as_matrix()
    ).to(transforms.device, dtype=transforms.dtype)
    
    # 4. Recompose the final 4x4 averaged transformation matrix
    T_align_avg = torch.eye(4, device=transforms.device, dtype=transforms.dtype)
    T_align_avg[:3, :3] = avg_rot_matrix
    T_align_avg[:3, 3] = avg_translation
    
    return T_align_avg