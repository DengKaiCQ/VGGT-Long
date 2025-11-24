import torch
import numpy as np

def weighted_estimate_se3_torch(source_points, target_points, weights):
    source_points = torch.from_numpy(source_points).cuda().float()
    target_points = torch.from_numpy(target_points).cuda().float()
    weights = torch.from_numpy(weights).cuda().float()
    
    total_weight = torch.sum(weights)
    if total_weight < 1e-6:
        return 1.0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)
    
    normalized_weights = weights / total_weight
    
    mu_src = torch.sum(normalized_weights[:, None] * source_points, dim=0)
    mu_tgt = torch.sum(normalized_weights[:, None] * target_points, dim=0)
    
    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt
    
    weighted_src = src_centered * torch.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * torch.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt
    
    return 1.0, mu_src.cpu().numpy(), mu_tgt.cpu().numpy(), H.cpu().numpy()

def weighted_estimate_sim3_torch(source_points, target_points, weights):

    source_points = torch.from_numpy(source_points).cuda().float()
    target_points = torch.from_numpy(target_points).cuda().float()
    weights = torch.from_numpy(weights).cuda().float()
    
    total_weight = torch.sum(weights)
    if total_weight < 1e-6:
        return -1.0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)
    
    normalized_weights = weights / total_weight
    
    mu_src = torch.sum(normalized_weights[:, None] * source_points, dim=0)
    mu_tgt = torch.sum(normalized_weights[:, None] * target_points, dim=0)
    
    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt
    
    scale_src = torch.sqrt(torch.sum(normalized_weights * torch.sum(src_centered**2, dim=1)))
    scale_tgt = torch.sqrt(torch.sum(normalized_weights * torch.sum(tgt_centered**2, dim=1)))
    s = scale_tgt / scale_src
    
    weighted_src = (s * src_centered) * torch.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * torch.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt
    
    return s.cpu().numpy(), mu_src.cpu().numpy(), mu_tgt.cpu().numpy(), H.cpu().numpy()

def weighted_estimate_sim3_numba_torch(source_points, target_points, weights, align_method='sim3'):

    if align_method == 'sim3':
        s, mu_src, mu_tgt, H = weighted_estimate_sim3_torch(source_points, target_points, weights)
    elif align_method == 'se3' or align_method == 'scale+se3':
        s, mu_src, mu_tgt, H = weighted_estimate_se3_torch(source_points, target_points, weights)
    
    if s < 0:
        raise ValueError("Total weight too small for meaningful estimation")
    
    H_torch = torch.from_numpy(H).cuda().float()
    U, _, Vt = torch.linalg.svd(H_torch)
    
    U = U.cpu().numpy()
    Vt = Vt.cpu().numpy()
    
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    mu_src = mu_src.astype(np.float32)
    mu_tgt = mu_tgt.astype(np.float32)
    R = R.astype(np.float32)
    
    if align_method == 'se3' or align_method == 'scale+se3':
        t = mu_tgt - R @ mu_src
    else:
        t = mu_tgt - s * R @ mu_src
    
    return s, R, t.astype(np.float32)

def huber_loss_torch(r, delta):

    r_torch = torch.from_numpy(r).cuda().float()
    delta_torch = torch.tensor(delta, device='cuda', dtype=torch.float32)
    
    abs_r = torch.abs(r_torch)
    result = torch.where(abs_r <= delta_torch,
                        0.5 * r_torch**2,
                        delta_torch * (abs_r - 0.5 * delta_torch))
    
    return result.cpu().numpy()

def compute_residuals_torch(tgt, transformed):

    tgt_torch = torch.from_numpy(tgt).cuda().float()
    transformed_torch = torch.from_numpy(transformed).cuda().float()
    
    residuals = torch.sqrt(torch.sum((tgt_torch - transformed_torch)**2, dim=1))
    return residuals.cpu().numpy()

def compute_huber_weights_torch(residuals, delta):

    residuals_torch = torch.from_numpy(residuals).cuda().float()
    delta_torch = torch.tensor(delta, device='cuda', dtype=torch.float32)
    
    weights = torch.ones_like(residuals_torch)
    mask = residuals_torch > delta_torch
    weights[mask] = delta_torch / residuals_torch[mask]
    
    return weights.cpu().numpy()

def apply_transformation_torch(src, s, R, t):

    src_torch = torch.from_numpy(src).cuda().float()
    R_torch = torch.from_numpy(R).cuda().float()
    t_torch = torch.from_numpy(t).cuda().float()
    s_torch = torch.tensor(s, device='cuda', dtype=torch.float32)
    
    transformed = s_torch * (src_torch @ R_torch.T) + t_torch
    return transformed.cpu().numpy()

def robust_weighted_estimate_sim3_torch(src, tgt, init_weights, delta=0.1, max_iters=20, tol=1e-9, align_method='sim3'):

    src = src.astype(np.float32)
    tgt = tgt.astype(np.float32)
    init_weights = init_weights.astype(np.float32)
    
    s, R, t = weighted_estimate_sim3_numba_torch(src, tgt, init_weights, align_method=align_method)
    
    prev_error = float('inf')
    
    for iter in range(max_iters):
        transformed = apply_transformation_torch(src, s, R, t)
        residuals = compute_residuals_torch(tgt, transformed)
        
        print(f'Iter {iter}: Mean residual = {np.mean(residuals):.6f}')
        
        huber_weights = compute_huber_weights_torch(residuals, delta)
        combined_weights = init_weights * huber_weights
        combined_weights /= (np.sum(combined_weights) + 1e-12)
        
        s_new, R_new, t_new = weighted_estimate_sim3_numba_torch(src, tgt, combined_weights, align_method=align_method)
        
        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1)/2)))
        
        current_error = np.sum(huber_loss_torch(residuals, delta) * init_weights)
        
        if (param_change < tol and rot_angle < np.radians(0.1)) or \
           (abs(prev_error - current_error) < tol * prev_error):
            print(f'Converged at iteration {iter}')
            break
        
        s, R, t = s_new, R_new, t_new
        prev_error = current_error
    
    return s, R, t

def warmup_torch():

    print("\nWarming up PyTorch alignment...")
    
    src = np.random.randn(100000, 3).astype(np.float32)
    tgt = np.random.randn(100000, 3).astype(np.float32)
    weights = np.ones(100000, dtype=np.float32)
    residuals = np.abs(np.random.randn(100000).astype(np.float32))
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    s = np.float32(1.0)
    delta = np.float32(1.0)
    
    try:
        _ = weighted_estimate_sim3_torch(src, tgt, weights)
        print(" - weighted_estimate_sim3_torch warmed up.")
    except Exception as e:
        print(" ! Failed to warm up weighted_estimate_sim3_torch:", e)
    
    try:
        _ = weighted_estimate_se3_torch(src, tgt, weights)
        print(" - weighted_estimate_se3_torch warmed up.")
    except Exception as e:
        print(" ! Failed to warm up weighted_estimate_se3_torch:", e)
    
    try:
        _ = huber_loss_torch(residuals, delta)
        print(" - huber_loss_torch warmed up.")
    except Exception as e:
        print(" ! Failed to warm up huber_loss_torch:", e)
    
    try:
        _ = compute_huber_weights_torch(residuals, delta)
        print(" - compute_huber_weights_torch warmed up.")
    except Exception as e:
        print(" ! Failed to warm up compute_huber_weights_torch:", e)
    
    try:
        _ = compute_residuals_torch(tgt, src)
        print(" - compute_residuals_torch warmed up.")
    except Exception as e:
        print(" ! Failed to warm up compute_residuals_torch:", e)
    
    try:
        _ = apply_transformation_torch(src, s, R, t)
        print(" - apply_transformation_torch warmed up.")
    except Exception as e:
        print(" ! Failed to warm up apply_transformation_torch:", e)
    
    print("PyTorch warm-up complete.\n")

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

if __name__ == "__main__":

    warmup_torch()
    
    n_points = 7_500_000
    src = np.random.randn(n_points, 3).astype(np.float32)
    
    true_R = np.array([[0.866, -0.5, 0],
                      [0.5, 0.866, 0],
                      [0, 0, 1]], dtype=np.float32)
    true_t = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    true_s = 1.2
    
    tgt = true_s * (src @ true_R.T) + true_t
    tgt += 0.01 * np.random.randn(*tgt.shape).astype(np.float32)
    
    weights = np.ones(n_points, dtype=np.float32)
    
    print_gpu_memory()
    
    s, R, t = robust_weighted_estimate_sim3_torch(
        src, tgt, weights, 
        delta=0.1, max_iters=5, align_method='sim3'
    )
    
    print(f"\nEstimated scale: {s:.6f}")
    print(f"Estimated rotation:\n{R}")
    print(f"Estimated translation: {t}")
    
    print_gpu_memory()