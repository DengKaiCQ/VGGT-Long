# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/tools/pose_interp.py

import torch

def decompose_se3_poses(poses_4x4):
    """
    Decompose (N, 4, 4) SE(3) poses into rotation matrices and translation vectors.
    
    Args:
        poses_4x4: (N, 4, 4) tensor
        
    Returns:
        R_matrices: (N, 3, 3) rotation matrices
        translations: (N, 3) translation vectors
    """
    R_matrices = poses_4x4[:, :3, :3]
    translations = poses_4x4[:, :3, 3]
    return R_matrices, translations

def recompose_se3_pose(R, t):
    """
    Recompose (N, 3, 3) rotation matrices and (N, 3) translation vectors into (N, 4, 4) SE(3) poses.
    
    Args:
        R: (N, 3, 3) rotation matrices
        t: (N, 3) translation vectors
        
    Returns:
        poses: (N, 4, 4) SE(3) poses
    """
    # Handle single pose case
    if R.dim() == 2:
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
        
    batch_size = R.shape[0]
    poses = torch.eye(4, dtype=R.dtype, device=R.device).repeat(batch_size, 1, 1)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = t
    
    return poses.squeeze(0) if batch_size == 1 else poses

def slerp_torch(q0, q1, t, DOT_THRESHOLD=0.9995):
    """
    Spherical Linear Interpolation (SLERP) for batch of quaternions using PyTorch.
    
    Args:
        q0: (N, 4) starting quaternions
        q1: (N, 4) ending quaternions
        t: (N,) or (N, 1) interpolation weights in [0, 1]
        DOT_THRESHOLD: Threshold for deciding when to use linear interpolation
        
    Returns:
        interpolated_quaternions: (N, 4)
    """
    # Ensure t is broadcastable
    if isinstance(t, float):
        t = torch.tensor(t, dtype=q0.dtype, device=q0.device)
    if t.dim() == 0:
        t = t.view(1, 1)
    elif t.dim() == 1:
        t = t.view(-1, 1)

    # Normalize inputs
    q0 = q0 / torch.norm(q0, dim=-1, keepdim=True)
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 represent the same rotation.
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.where(dot < 0, -dot, dot)

    # Clamp dot product to be within [-1, 1] to avoid nan in acos
    dot = torch.clamp(dot, -1.0, 1.0)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    # Use linear interpolation when angles are very small
    use_lerp = dot > DOT_THRESHOLD
    
    s0 = torch.where(use_lerp, 1.0 - t, s0)
    s1 = torch.where(use_lerp, t, s1)
    
    res = s0 * q0 + s1 * q1
    res = res / torch.norm(res, dim=-1, keepdim=True)
    
    return res

def matrix_to_quaternion(matrix):
    """
    Convert 3x3 rotation matrices to quaternions.
    Based on PyTorch3D implementation.
    
    Args:
        matrix: (N, 3, 3) rotation matrices
    
    Returns:
        quaternions: (N, 4) [w, x, y, z]
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    
    o0 = 0.5 * torch.sqrt(torch.abs(1 + m00 + m11 + m22))
    x = 0.5 * torch.sqrt(torch.abs(1 + m00 - m11 - m22))
    y = 0.5 * torch.sqrt(torch.abs(1 - m00 + m11 - m22))
    z = 0.5 * torch.sqrt(torch.abs(1 - m00 - m11 + m22))
    
    # o1 = torch.where(m21 > m12, x, -x) # type: ignore
    # o2 = torch.where(m02 > m20, y, -y) # type: ignore
    # o3 = torch.where(m10 > m01, z, -z) # type: ignore
    
    # Needs a more robust implementation for general cases, 
    # use a simplified trace method for now or importing a library is better.
    # However, to be self-contained and robust, let's use a standard implementation.
    # Below is a more robust trace method.
    
    trace = m00 + m11 + m22
    cond = trace > 0
    
    # Trace > 0
    s_cond = 0.5 / torch.sqrt(trace + 1.0)
    w_cond = 0.25 / s_cond
    x_cond = (matrix[..., 2, 1] - matrix[..., 1, 2]) * s_cond
    y_cond = (matrix[..., 0, 2] - matrix[..., 2, 0]) * s_cond
    z_cond = (matrix[..., 1, 0] - matrix[..., 0, 1]) * s_cond
    
    # Trace <= 0
    # Find max diagonal element
    d0 = matrix[..., 0, 0]
    d1 = matrix[..., 1, 1]
    d2 = matrix[..., 2, 2]
    
    cond_0 = (d0 > d1) & (d0 > d2)
    cond_1 = (d1 > d0) & (d1 > d2)
    # cond_2 is remainder
    
    # Case 0
    s0 = 2.0 * torch.sqrt(1.0 + d0 - d1 - d2)
    w0 = (matrix[..., 2, 1] - matrix[..., 1, 2]) / s0
    x0 = 0.25 * s0
    y0 = (matrix[..., 0, 1] + matrix[..., 1, 0]) / s0
    z0 = (matrix[..., 0, 2] + matrix[..., 2, 0]) / s0

    # Case 1
    s1 = 2.0 * torch.sqrt(1.0 + d1 - d0 - d2)
    w1 = (matrix[..., 0, 2] - matrix[..., 2, 0]) / s1
    x1 = (matrix[..., 0, 1] + matrix[..., 1, 0]) / s1
    y1 = 0.25 * s1
    z1 = (matrix[..., 1, 2] + matrix[..., 2, 1]) / s1
    
    # Case 2
    s2 = 2.0 * torch.sqrt(1.0 + d2 - d0 - d1)
    w2 = (matrix[..., 1, 0] - matrix[..., 0, 1]) / s2
    x2 = (matrix[..., 0, 2] + matrix[..., 2, 0]) / s2
    y2 = (matrix[..., 1, 2] + matrix[..., 2, 1]) / s2
    z2 = 0.25 * s2
    
    # Combine
    w = torch.where(cond, w_cond, torch.where(cond_0, w0, torch.where(cond_1, w1, w2)))
    x = torch.where(cond, x_cond, torch.where(cond_0, x0, torch.where(cond_1, x1, x2)))
    y = torch.where(cond, y_cond, torch.where(cond_0, y0, torch.where(cond_1, y1, y2)))
    z = torch.where(cond, z_cond, torch.where(cond_0, z0, torch.where(cond_1, z1, z2)))
    
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    
    Args:
        quaternions: (N, 4) tensor with quaternions
    
    Returns:
        outputs: (N, 3, 3) rotation matrices
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def interpolate_poses(poses_a, poses_b, confidence_a=None, confidence_b=None, interpolate=True):
    """
    Interpolate between two sets of poses based on confidence scores.

    Args:
        poses_a: (N, 4, 4) tensor of poses
        poses_b: (N, 4, 4) tensor of poses
        confidence_a: (N, H, W) tensor of confidence maps for poses_a
        confidence_b: (N, H, W) tensor of confidence maps for poses_b
        interpolate: bool, whether to interpolate or choose max confidence

    Returns:
        merged_poses: (N, 4, 4) tensor of interpolated poses
    """
    # poses_a, poses_b: (overlap_len, 4, 4)
    # confidence_a, confidence_b: (overlap_len, H, W)
    
    device = poses_a.device
    overlap_len = poses_a.shape[0]

    # Calculate mean confidence for each pose
    # Default to uniform confidence if not provided
    conf_a_mean = confidence_a.mean(dim=(1,2)) if confidence_a is not None else torch.ones(overlap_len, device=device)
    conf_b_mean = confidence_b.mean(dim=(1,2)) if confidence_b is not None else torch.ones(overlap_len, device=device)

    # Decompose poses
    R1, t1 = decompose_se3_poses(poses_a)
    R2, t2 = decompose_se3_poses(poses_b)

    if not interpolate:
        # Return the one with higher TOTAL confidence across all frames? 
        # The original code did: conf1_mean.mean() >= conf2_mean.mean() which is a single boolean for the whole batch.
        # This seems to mean "pick the best trajectory".
        if conf_a_mean.mean() >= conf_b_mean.mean():
            return poses_a
        else:
            return poses_b

    # Vectorized interpolation
    total_conf = conf_a_mean + conf_b_mean
    # Avoid division by zero
    total_conf = torch.clamp(total_conf, min=1e-6)
    
    alpha = conf_b_mean / total_conf # (N,)
    alpha = alpha.unsqueeze(-1) # (N, 1)

    # Interpolate Translation (Linear)
    # t: (N, 3)
    t_merged = (1 - alpha) * t1 + alpha * t2
    
    # Interpolate Rotation (SLERP)
    # Convert matrices to quaternions
    q1 = matrix_to_quaternion(R1)
    q2 = matrix_to_quaternion(R2)
    
    q_merged = slerp_torch(q1, q2, alpha.squeeze(-1))
    
    # Convert back to matrix
    R_merged = quaternion_to_matrix(q_merged)
    
    # Recompose
    pose_merged = recompose_se3_pose(R_merged, t_merged)
    
    return pose_merged
