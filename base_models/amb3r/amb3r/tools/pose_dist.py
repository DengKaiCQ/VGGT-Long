# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/tools/pose_dist.py

import numpy as np

def rotation_angle(R1, R2):
    # R1 and R2 are 3x3 rotation matrices
    R = R1.T @ R2
    # Numerical stability: clamp values into [-1,1]
    val = (np.trace(R) - 1) / 2
    val = np.clip(val, -1.0, 1.0)
    angle_rad = np.arccos(val)
    angle_deg = np.degrees(angle_rad)  # Convert radians to degrees
    return angle_deg

def extrinsic_distance(extrinsic1, extrinsic2, lambda_t=1.0):
    R1, t1 = extrinsic1[:3, :3], extrinsic1[:3, 3]
    R2, t2 = extrinsic2[:3, :3], extrinsic2[:3, 3]
    rot_diff = rotation_angle(R1, R2) / 180
    
    center_diff = np.linalg.norm(t1 - t2)
    return rot_diff + lambda_t * center_diff

def rotation_angle_batch(R1, R2):
    # R1, R2: shape (N, 3, 3)
    # We want a matrix of rotation angles for all pairs.
    # We'll get R1^T R2 for each pair.
    # Expand dimensions to broadcast: 
    # R1^T: (N,3,3) -> (N,1,3,3)
    # R2: (N,3,3) -> (1,N,3,3)
    R1_t = np.transpose(R1, (0, 2, 1))[:, np.newaxis, :, :]  # shape (N,1,3,3)
    R2_b = R2[np.newaxis, :, :, :]                          # shape (1,N,3,3)
    R_mult = np.matmul(R1_t, R2_b)  # shape (N,N,3,3)
    # trace(R) for each pair
    trace_vals = R_mult[..., 0, 0] + R_mult[..., 1, 1] + R_mult[..., 2, 2]  # (N,N)
    val = (trace_vals - 1) / 2
    val = np.clip(val, -1.0, 1.0)
    angle_rad = np.arccos(val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg / 180.0  # normalized rotation difference
def extrinsic_distance_batch(extrinsics, lambda_t=1.0):
    # extrinsics: (N,4,4)
    # Extract rotation and translation
    R = extrinsics[:, :3, :3]  # (N,3,3)
    t = extrinsics[:, :3, 3]   # (N,3)
    # Compute all pairwise rotation differences
    rot_diff = rotation_angle_batch(R, R)  # (N,N)
    # Compute all pairwise translation differences
    # For t, shape (N,3). We want all pair differences: t[i] - t[j].
    # t_i: (N,1,3), t_j: (1,N,3)
    t_i = t[:, np.newaxis, :]  # (N,1,3)
    t_j = t[np.newaxis, :, :]  # (1,N,3)
    trans_diff = np.linalg.norm(t_i - t_j, axis=2)  # (N,N)
    dists = rot_diff + lambda_t * trans_diff
    return dists

def extrinsic_distance_batch_query(extrinsics1, extrinsics2, lambda_t=1.0):
    # extrinsics: (N,4,4)
    # Extract rotation and translation
    R1 = extrinsics1[:, :3, :3]  # (N,3,3)
    t1 = extrinsics1[:, :3, 3]   # (N,3)
    R2 = extrinsics2[:, :3, :3]  # (M,3,3)
    t2 = extrinsics2[:, :3, 3]   # (M,
    # Compute all pairwise rotation differences
    rot_diff = rotation_angle_batch(R1, R2)  # (N,N)
    # Compute all pairwise translation differences
    # For t, shape (N,3). We want all pair differences: t[i] - t[j].
    # t_i: (N,1,3), t_j: (1,N,3)
    t_i = t1[:, np.newaxis, :]  # (N,1,3)
    t_j = t2[np.newaxis, :, :]  # (1,N,3)
    trans_diff = np.linalg.norm(t_i - t_j, axis=2)  # (N,N)
    dists = rot_diff + lambda_t * trans_diff
    return dists

def rotation_angle_batch_chunked(R, chunk_size):
    N = R.shape[0]
    rot_diff = np.empty((N, N), dtype=np.float32)
    # Precompute R transpose once
    R_t = R.transpose(0,2,1)
    
    for i_start in range(0, N, chunk_size):
        i_end = min(N, i_start + chunk_size)
        # Sub-block of R_t
        R_i_t = R_t[i_start:i_end]  # (B,3,3)
        
        for j_start in range(0, N, chunk_size):
            j_end = min(N, j_start + chunk_size)
            R_j = R[j_start:j_end]   # (B,3,3)
            # Compute R_i_t @ R_j for block
            # R_i_t: (B,3,3)
            # R_j:   (B,3,3) but we need pairwise, so we expand dims
            # This still can be large. If even BxB is too big, choose smaller chunks.
            # shape (B,B,3,3)
            R_mult = R_i_t[:, np.newaxis, :, :] @ R_j[np.newaxis, :, :, :]
            # Compute trace
            trace_vals = R_mult[...,0,0] + R_mult[...,1,1] + R_mult[...,2,2]
            val = (trace_vals - 1.0) / 2.0
            np.clip(val, -1.0, 1.0, out=val)
            angle_rad = np.arccos(val)
            angle_deg = np.degrees(angle_rad)
            block_rot_diff = angle_deg / 180.0
            rot_diff[i_start:i_end, j_start:j_end] = block_rot_diff.astype(np.float32)
    return rot_diff

def extrinsic_distance_batch_chunked(extrinsics, lambda_t=1.0, chunk_size=1000):
    R = extrinsics[:, :3, :3].astype(np.float32)
    t = extrinsics[:, :3, 3].astype(np.float32)
    N = R.shape[0]
    # Compute rotation differences in chunks
    rot_diff = rotation_angle_batch_chunked(R, chunk_size)
    # Compute translation differences in chunks
    dists = np.empty((N, N), dtype=np.float32)
    for i_start in range(0, N, chunk_size):
        i_end = min(N, i_start + chunk_size)
        t_i = t[i_start:i_end]  # (B,3)
        for j_start in range(0, N, chunk_size):
            j_end = min(N, j_start + chunk_size)
            t_j = t[j_start:j_end]  # (B,3)
            
            # broadcasting: (B,1,3) - (1,B,3) => (B,B,3)
            diff = t_i[:, None, :] - t_j[None, :, :]
            trans_diff = np.linalg.norm(diff, axis=2)  # (B,B)
            
            # Add rotation and translation
            dists[i_start:i_end, j_start:j_end] = rot_diff[i_start:i_end, j_start:j_end] + lambda_t * trans_diff
    return dists


def compute_ranking(extrinsics, lambda_t=1.0, normalize=True, batched=True):
    
    if normalize:
        extrinsics = np.copy(extrinsics)
        camera_center = np.copy(extrinsics[:, :3, 3])
        camera_center_scale = np.linalg.norm(camera_center, axis=1)
        avg_scale = np.mean(camera_center_scale)
        extrinsics[:, :3, 3] = extrinsics[:, :3, 3] / avg_scale
    
    
    if batched:
        if len(extrinsics) > 6000:
            dists = extrinsic_distance_batch_chunked(extrinsics, lambda_t=lambda_t)
        else:
            dists = extrinsic_distance_batch(extrinsics, lambda_t=lambda_t)
    else:
        N = extrinsics.shape[0]
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i,j] = extrinsic_distance(extrinsics[i], extrinsics[j], lambda_t=lambda_t)
    ranking = np.argsort(dists, axis=1)
    return ranking, dists
