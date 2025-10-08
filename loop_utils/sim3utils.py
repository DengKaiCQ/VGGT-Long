
import os
import numpy as np
import trimesh

from numba import njit

import glob
import bisect

def accumulate_sim3_transforms(transforms):
    """
    Accumulate adjacent SIM(3) transforms into transforms from the initial frame to each subsequent frame.
    
    Args:
    transforms: list, each element is a tuple (R, s, t)
        R: 3x3 rotation matrix (np.array)
        s: scale factor (scalar)
        t: 3x1 translation vector (np.array)
    
    Returns:
    Cumulative transforms list, each element is (R_cum, s_cum, t_cum)
        representing the transform from frame 0 to frame k
    """
    if not transforms:
        return []
    
    cumulative_transforms = [transforms[0]]
    
    for i in range(1, len(transforms)):
        s_cum_prev, R_cum_prev, t_cum_prev = cumulative_transforms[i-1]
        s_next, R_next, t_next = transforms[i]
        R_cum_new = R_cum_prev @ R_next
        s_cum_new = s_cum_prev * s_next
        t_cum_new = s_cum_prev * (R_cum_prev @ t_next) + t_cum_prev
        cumulative_transforms.append((s_cum_new, R_cum_new, t_cum_new))
    
    return cumulative_transforms

def estimate_sim3(source_points, target_points):
    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt((src_centered ** 2).sum(axis=1).mean())
    scale_tgt = np.sqrt((tgt_centered ** 2).sum(axis=1).mean())
    s = scale_tgt / scale_src

    src_scaled = src_centered * s

    H = src_scaled.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t

def align_point_maps(point_map1, conf1, point_map2, conf2, conf_threshold):
    """ point_map2 -> point_map1"""
    b1, _, _, _ = point_map1.shape
    b2, _, _, _ = point_map2.shape
    b = min(b1, b2)
    
    aligned_points1 = []
    aligned_points2 = []

    for i in range(b):
        mask1 = conf1[i] > conf_threshold
        mask2 = conf2[i] > conf_threshold
        valid_mask = mask1 & mask2

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        pts1 = point_map1[i][idx]
        pts2 = point_map2[i][idx]

        aligned_points1.append(pts1)
        aligned_points2.append(pts2)

    if len(aligned_points1) == 0:
        raise ValueError("No matching point pairs were found!")

    all_pts1 = np.concatenate(aligned_points1, axis=0)
    all_pts2 = np.concatenate(aligned_points2, axis=0)

    print(f"The number of corresponding points matched: {all_pts1.shape[0]}")
    s, R, t = estimate_sim3(all_pts2, all_pts1)

    mean_error = compute_alignment_error(
        point_map1, conf1, 
        point_map2, conf2, 
        conf_threshold, 
        s, R, t
    )
    print(f'Mean error: {mean_error}')

    return s, R, t

def apply_sim3(points, s, R, t):
    return (s * (R @ points.T)).T + t


def apply_sim3_direct(point_maps, s, R, t):
    # point_maps: (b, h, w, 3) -> (b, h, w, 3, 1)
    point_maps_expanded = point_maps[..., np.newaxis]  # (b, h, w, 3, 1)

    # R: (3, 3) -> (b, h, w, 3, 1) = (3, 3) @ (3, 1)
    rotated = np.matmul(R, point_maps_expanded)  # (b, h, w, 3, 1)
    rotated = rotated.squeeze(-1)  # (b, h, w, 3)
    transformed = s * rotated + t  # (b, h, w, 3)

    return transformed


def compute_alignment_error(point_map1, conf1, point_map2, conf2, conf_threshold, s, R, t):
    """
    Compute the average point alignment error (using only original inputs)
    
    Args:
    point_map1: target point map (b, h, w, 3)
    conf1: target confidence map (b, h, w)
    point_map2: source point map (b, h, w, 3)
    conf2: source confidence map (b, h, w)
    conf_threshold: confidence threshold
    s, R, t: transformation parameters
    """
    b1, h1, w1, _ = point_map1.shape
    b2, h2, w2, _ = point_map2.shape
    b = min(b1, b2)
    h = min(h1, h2)
    w = min(w1, w2)
    
    target_points = []
    source_points = []
    
    for i in range(b):
        mask1 = conf1[i, :h, :w] > conf_threshold
        mask2 = conf2[i, :h, :w] > conf_threshold
        valid_mask = mask1 & mask2

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue
            
        t_pts = point_map1[i, :h, :w][idx]
        s_pts = point_map2[i, :h, :w][idx]
        
        target_points.append(t_pts)
        source_points.append(s_pts)
    
    if len(target_points) == 0:
        print("Warning: No matching point pairs found for error calculation")
        return np.nan
    
    all_target = np.concatenate(target_points, axis=0)
    all_source = np.concatenate(source_points, axis=0)
    
    transformed = (s * (R @ all_source.T)).T + t
    
    errors = np.linalg.norm(transformed - all_target, axis=1)
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)
    
    print(f"Alignment error statistics [using {len(errors)} points]: "
          f"mean={mean_error:.4f}, std={std_error:.4f}, "
          f"median={median_error:.4f}, max={max_error:.4f}")
    
    return mean_error



def save_confident_pointcloud(points, colors, confs, output_path, conf_threshold, sample_ratio=1.0):
    """
    Filter points based on confidence threshold and save as PLY file, with optional random sampling ratio.

    Args:
    - points: np.ndarray, shape (H, W, 3) or (N, 3)
    - colors: np.ndarray, shape (H, W, 3) or (N, 3)
    - confs: np.ndarray, shape (H, W) or (N,)
    - output_path: str, output PLY file path
    - conf_threshold: float, confidence threshold for point filtering
    - sample_ratio: float, sampling ratio (0 < sample_ratio <= 1.0)
    """
    points = points.reshape(-1, 3).astype(np.float32, copy=False)
    colors = colors.reshape(-1, 3).astype(np.uint8, copy=False)
    confs = confs.reshape(-1).astype(np.float32, copy=False)

    conf_mask = (confs >= conf_threshold) & (confs > 1e-5)
    points = points[conf_mask]
    colors = colors[conf_mask]

    if 0 < sample_ratio < 1.0 and len(points) > 0:
        num_samples = int(len(points) * sample_ratio)
        indices = np.random.choice(len(points), num_samples, replace=False)
        points = points[indices]
        colors = colors[indices]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f'shape of sampled point: {points.shape}')
    trimesh.PointCloud(points, colors=colors).export(output_path)
    print(f"Saved point cloud with {len(points)} points to {output_path}")


def save_confident_pointcloud_batch(points, colors, confs, output_path, conf_threshold, sample_ratio=1.0, batch_size=1000000):
    """
    - points: np.ndarray,  (b, H, W, 3) / (N, 3)
    - colors: np.ndarray,  (b, H, W, 3) / (N, 3)
    - confs: np.ndarray,  (b, H, W) / (N,)
    - output_path: str
    - conf_threshold: float,
    - sample_ratio: float (0 < sample_ratio <= 1.0)
    - batch_size: int
    """
    if points.ndim == 2:
        b = 1
        points = points[np.newaxis, ...]
        colors = colors[np.newaxis, ...]
        confs = confs[np.newaxis, ...]
    elif points.ndim == 4:
        b = points.shape[0]
    else:
        raise ValueError("Unsupported points dimension. Must be 2 (N,3) or 4 (b,H,W,3)")
    
    total_valid = 0
    for i in range(b):
        cfs = confs[i].reshape(-1)
        total_valid += np.count_nonzero((cfs >= conf_threshold) & (cfs > 1e-5))
    
    num_samples = int(total_valid * sample_ratio) if sample_ratio < 1.0 else total_valid
    
    if num_samples == 0:
        save_ply(np.zeros((0, 3), dtype=np.float32), 
                 np.zeros((0, 3), dtype=np.uint8), 
                 output_path)
        return
    
    if sample_ratio == 1.0:
        with open(output_path, 'wb') as f:
            write_ply_header(f, num_samples)
            
            for i in range(b):
                pts = points[i].reshape(-1, 3).astype(np.float32)
                cls = colors[i].reshape(-1, 3).astype(np.uint8)
                cfs = confs[i].reshape(-1).astype(np.float32)
                
                mask = (cfs >= conf_threshold) & (cfs > 1e-5)
                valid_pts = pts[mask]
                valid_cls = cls[mask]
                
                for j in range(0, len(valid_pts), batch_size):
                    batch_pts = valid_pts[j:j+batch_size]
                    batch_cls = valid_cls[j:j+batch_size]
                    write_ply_batch(f, batch_pts, batch_cls)
    
    else:
        reservoir_pts = np.zeros((num_samples, 3), dtype=np.float32)
        reservoir_clr = np.zeros((num_samples, 3), dtype=np.uint8)
        count = 0
        
        for i in range(b):
            pts = points[i].reshape(-1, 3).astype(np.float32)
            cls = colors[i].reshape(-1, 3).astype(np.uint8)
            cfs = confs[i].reshape(-1).astype(np.float32)
            
            mask = (cfs >= conf_threshold) & (cfs > 1e-5)
            valid_pts = pts[mask]
            valid_cls = cls[mask]
            n_valid = len(valid_pts)
            
            if count < num_samples:
                fill_count = min(num_samples - count, n_valid)
                
                reservoir_pts[count:count+fill_count] = valid_pts[:fill_count]
                reservoir_clr[count:count+fill_count] = valid_cls[:fill_count]
                count += fill_count
                
                if fill_count < n_valid:
                    remaining_pts = valid_pts[fill_count:]
                    remaining_cls = valid_cls[fill_count:]
                    
                    count, reservoir_pts, reservoir_clr = optimized_vectorized_reservoir_sampling(
                        remaining_pts, remaining_cls, count, reservoir_pts, reservoir_clr
                    )
            else:
                count, reservoir_pts, reservoir_clr = optimized_vectorized_reservoir_sampling(
                    valid_pts, valid_cls, count, reservoir_pts, reservoir_clr
                )
        
        save_ply(reservoir_pts, reservoir_clr, output_path)


### ========= ###

''' The following function is deprecated'''

# def vectorized_reservoir_sampling(new_pts, new_cls, current_count, reservoir_pts, reservoir_clr):
#     """
#     - new_pts:  (M, 3)
#     - new_cls:  (M, 3)
#     - current_count
#     - reservoir_pts:  (K, 3)
#     - reservoir_clr:  (K, 3)
    
#     """
#     k = len(reservoir_pts)
#     n_new = len(new_pts)
    
#     rand_indices = np.random.randint(0, current_count + n_new, size=n_new)
    
#     replace_mask = rand_indices < k
#     replace_indices = rand_indices[replace_mask]
#     replace_pts = new_pts[replace_mask]
#     replace_cls = new_cls[replace_mask]
    
#     reservoir_pts[replace_indices] = replace_pts
#     reservoir_clr[replace_indices] = replace_cls
    
#     return current_count + n_new, reservoir_pts, reservoir_clr

### ========= ###


'''
    Function `vectorized_reservoir_sampling`  is not mathematically accurate in sampling.
    This leads to inconsistent density in the downsampled point clouds. 
    The `optimized_vectorized_reservoir_sampling` function has fixed this bug.

    Special thanks to @Horace89 for the detailed analysis and code assistance.

    See https://github.com/DengKaiCQ/VGGT-Long/issues/28 for details
'''

def optimized_vectorized_reservoir_sampling(
    new_points: np.ndarray,
    new_colors: np.ndarray,
    current_count: int, 
    reservoir_points: np.ndarray,
    reservoir_colors: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Optimized vectorized reservoir sampling with batch probability calculations.
    
    This maintains mathematical correctness while improving performance through
    vectorized operations where possible.
    
    Args:
        new_points: New point coordinates to consider, shape (M, 3)
        new_colors: New point colors to consider, shape (M, 3)
        current_count: Number of elements seen so far  
        reservoir_points: Current reservoir of sampled points, shape (K, 3)
        reservoir_colors: Current reservoir of sampled colors, shape (K, 3)
        
    Returns:
        Tuple of (updated_count, updated_reservoir_points, updated_reservoir_colors)
    """
    random_gen = np.random
        
    reservoir_size = len(reservoir_points)
    num_new_points = len(new_points)
    
    if num_new_points == 0:
        return current_count, reservoir_points, reservoir_colors
    
    # Calculate sequential indices for each new point
    point_indices = np.arange(current_count + 1, current_count + num_new_points + 1)
    
    # Generate random numbers for each point
    random_values = random_gen.randint(0, point_indices, size=num_new_points)
    
    # Determine which points should replace reservoir elements
    replacement_mask = random_values < reservoir_size
    replacement_positions = random_values[replacement_mask]
    
    # Apply replacements
    if np.any(replacement_mask):
        points_to_replace = new_points[replacement_mask]
        colors_to_replace = new_colors[replacement_mask]
        
        reservoir_points[replacement_positions] = points_to_replace
        reservoir_colors[replacement_positions] = colors_to_replace
    
    return current_count + num_new_points, reservoir_points, reservoir_colors



def write_ply_header(f, num_vertices):
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    f.write("\n".join(header).encode() + b"\n")

def write_ply_batch(f, points, colors):
    structured = np.zeros(len(points), dtype=[
        ('x', np.float32), 
        ('y', np.float32), 
        ('z', np.float32),
        ('red', np.uint8),
        ('green', np.uint8),
        ('blue', np.uint8)
    ])
    
    structured['x'] = points[:, 0]
    structured['y'] = points[:, 1]
    structured['z'] = points[:, 2]
    structured['red'] = colors[:, 0]
    structured['green'] = colors[:, 1]
    structured['blue'] = colors[:, 2]
    
    f.write(structured.tobytes())

def save_ply(points, colors, filename):
    with open(filename, 'wb') as f:
        write_ply_header(f, len(points))
        write_ply_batch(f, points, colors)



def find_chunk_index(chunks, idx):
    """
    Find the 0-based chunk index that contains the given index idx.
    chunks: List of (begin_idx, end_idx).
    idx: The index to search for.
    Returns the 0-based chunk index.
    """
    starts = [chunk[0] for chunk in chunks]
    pos = bisect.bisect_right(starts, idx) - 1  # Find position of idx in starts
    if pos < 0 or pos >= len(chunks):
        raise ValueError(f"Index {idx} not found in any chunk")
    chunk_begin, chunk_end = chunks[pos]
    if idx < chunk_begin or idx > chunk_end:
        raise ValueError(f"Index {idx} not found in any chunk")
    return pos

def get_frame_range(chunk, idx, half_window=10):
    """
    Calculate the frame range centered at idx with half_window frames on each side within chunk boundaries.
    If near boundaries, take 2 * half_window frames starting from the boundary.
    chunk: (begin_idx, end_idx).
    idx: Center index.
    half_window: Number of frames to take on each side of center index.
    Returns (start, end).
    """
    begin, end = chunk
    window_size = 2 * half_window

    if idx - half_window < begin:
        start = begin
        end_candidate = begin + window_size
        end = min(end, end_candidate)

    elif idx + half_window > end:
        end_candidate = end
        start_candidate = end - window_size
        start = max(begin, start_candidate)

    else:
        start = idx - half_window
        end = idx + half_window
    return (start, end)

def process_loop_list(chunk_index, loop_list, half_window=10):
    """
    Process loop_list and return chunk indices and frame ranges for each (idx1, idx2) pair.
    chunk_index: List of (begin_idx, end_idx) tuples.
    loop_list: List of (idx1, idx2) tuples.
    half_window: Number of frames to take on each side of center index (default 10).
    Returns list of (chunk_idx1, range1, chunk_idx2, range2) tuples where:
      - chunk_idx1, chunk_idx2: Chunk indices (1-based).
      - range1, range2: Frame range tuples (start, end).
    """
    results = []
    for idx1, idx2 in loop_list:
        try:
            chunk_idx1_0based = find_chunk_index(chunk_index, idx1)
            chunk1 = chunk_index[chunk_idx1_0based]
            range1 = get_frame_range(chunk1, idx1, half_window)
            
            chunk_idx2_0based = find_chunk_index(chunk_index, idx2)
            chunk2 = chunk_index[chunk_idx2_0based]
            range2 = get_frame_range(chunk2, idx2, half_window)
            

            result = (
                chunk_idx1_0based,
                range1,
                chunk_idx2_0based,
                range2
            )
            results.append(result)
        except ValueError as e:
            print(f"Skipping pair ({idx1}, {idx2}): {e}")
    return results


def compute_sim3_ab(S_a, S_b):

    s_a, R_a, T_a = S_a
    s_b, R_b, T_b = S_b

    s_ab = s_b / s_a
    R_ab = R_b @ R_a.T
    T_ab = T_b - s_ab * (R_ab @ T_a)

    return (s_ab, R_ab, T_ab)


def merge_ply_files(input_dir, output_path):
    """
    Merge all PLY files in a directory into one file (without loading into memory)
    
    Args:
    - input_dir: Input directory containing multiple '{idx}_pcd.ply' files
    - output_path: Output file path (e.g., 'combined.ply')
    """

    input_files = sorted(glob.glob(os.path.join(input_dir, '*_pcd.ply')))
    
    if not input_files:
        print("No PLY files found")
        return
    
    idx_file = 0
    all_file_num = len(input_files)
    
    total_vertices = 0
    for file in input_files: # Count total vertices
        with open(file, 'rb') as f:
            for line in f:
                if line.startswith(b'element vertex'):
                    vertex_count = int(line.split()[-1])
                    total_vertices += vertex_count
                elif line.startswith(b'end_header'):
                    break
    
    with open(output_path, 'wb') as out_f:
        # Write new header
        out_f.write(b"ply\n")
        out_f.write(b"format binary_little_endian 1.0\n")
        out_f.write(f"element vertex {total_vertices}\n".encode())
        out_f.write(b"property float x\n")
        out_f.write(b"property float y\n")
        out_f.write(b"property float z\n")
        out_f.write(b"property uchar red\n")
        out_f.write(b"property uchar green\n")
        out_f.write(b"property uchar blue\n")
        out_f.write(b"end_header\n")
        
        for file in input_files:
            # print(f'{idx_file}/{all_file_num}')
            idx_file += 1
            with open(file, 'rb') as in_f:
                # Skip the head
                in_header = True
                while in_header:
                    line = in_f.readline()
                    if line.startswith(b'end_header'):
                        in_header = False
                
                chunk_size = 1024 * 1024 * 10  # 10MB
                while True:
                    chunk = in_f.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
    
    print(f"Merge completed! Total points: {total_vertices}")
    print(f"Output file: {output_path}")

def weighted_estimate_se3(source_points, target_points, weights):
    """
    source_points:  (Nx3)
    target_points:  (Nx3)
    :weights:  (N,) [0,1]
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")
    
    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    weighted_src = src_centered * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - R @ mu_src
    
    return 1.0, R, t

def weighted_estimate_sim3(source_points, target_points, weights):
    """
    source_points:  (Nx3)
    target_points:  (Nx3)
    :weights:  (N,) [0,1]
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")
    
    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
    scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
    s = scale_tgt / scale_src

    weighted_src = (s * src_centered) * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


def huber_loss(r, delta):
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))

def robust_weighted_estimate_sim3(src, tgt, init_weights, delta=0.1, max_iters=20, tol=1e-9, using_sim3 = True):
    """
    src:  (Nx3)
    tgt:  (Nx3)
    init_weights:  (N,)
    """
    if using_sim3:
        s, R, t = weighted_estimate_sim3(src, tgt, init_weights)
    else:
        s, R, t = weighted_estimate_se3(src, tgt, init_weights)
    prev_error = float('inf')
    
    for iter in range(max_iters):

        transformed = s * (src @ R.T) + t
        residuals = np.linalg.norm(tgt - transformed, axis=1)  # (N,)
        print(f'Residuals: {np.mean(residuals)}')
        
        abs_res = np.abs(residuals)
        huber_weights = np.ones_like(residuals)
        large_res_mask = abs_res > delta
        huber_weights[large_res_mask] = delta / abs_res[large_res_mask]
        
        combined_weights = init_weights * huber_weights
        
        combined_weights /= (np.sum(combined_weights) + 1e-12)
        
        if using_sim3:
            s_new, R_new, t_new = weighted_estimate_sim3(src, tgt, init_weights)
        else:
            s_new, R_new, t_new = weighted_estimate_se3(src, tgt, init_weights)

        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1)/2)))
        current_error = np.sum(huber_loss(residuals, delta) * init_weights)
        
        if (param_change < tol and rot_angle < np.radians(0.1)) or \
           (abs(prev_error - current_error) < tol * prev_error):
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error
    
    return s, R, t


# ===== Speed Up Begin =====

@njit(cache=True)
def _weighted_estimate_se3_numba(source_points, target_points, weights):
    # Ensure float32
    source_points = source_points.astype(np.float32)
    target_points = target_points.astype(np.float32)
    weights = weights.astype(np.float32)

    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        return 1.0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)

    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    weighted_src = src_centered * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]

    H = weighted_src.T @ weighted_tgt

    return 1.0, mu_src, mu_tgt, H

@njit(cache=True)
def _weighted_estimate_sim3_numba(source_points, target_points, weights):
    # Ensure float32
    source_points = source_points.astype(np.float32)
    target_points = target_points.astype(np.float32)
    weights = weights.astype(np.float32)

    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        return -1.0, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)

    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
    scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
    s = scale_tgt / scale_src

    weighted_src = (s * src_centered) * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]

    H = weighted_src.T @ weighted_tgt

    return s, mu_src, mu_tgt, H

def weighted_estimate_sim3_numba(source_points, target_points, weights, using_sim3 = True):
    if using_sim3:
        s, mu_src, mu_tgt, H = _weighted_estimate_sim3_numba(source_points, target_points, weights)
    else:
        s, mu_src, mu_tgt, H = _weighted_estimate_se3_numba(source_points, target_points, weights)

    if s < 0:
        raise ValueError("Total weight too small for meaningful estimation")

    # Ensure float32
    H = H.astype(np.float32)
    U, _, Vt = np.linalg.svd(H.astype(np.float32))  # float32 SVD

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    if using_sim3:
        t = mu_tgt - s * R @ mu_src
    else:
        t = mu_tgt - R @ mu_src

    return s, R, t

@njit(cache=True)
def huber_loss_numba(r, delta):
    r = r.astype(np.float32)
    delta = np.float32(delta)
    abs_r = np.abs(r)
    result = np.where(abs_r <= delta,
                      0.5 * r**2,
                      delta * (abs_r - 0.5 * delta))
    return result.astype(np.float32)

@njit(cache=True)
def compute_residuals_numba(tgt, transformed):
    residuals = np.empty(tgt.shape[0], dtype=np.float32)
    for i in range(tgt.shape[0]):
        diff = tgt[i] - transformed[i]
        residuals[i] = np.sqrt(np.sum(diff ** 2))
    return residuals

@njit(cache=True)
def compute_huber_weights_numba(residuals, delta):
    weights = np.ones(residuals.shape, dtype=np.float32)
    for i in range(residuals.shape[0]):
        r = residuals[i]
        if r > delta:
            weights[i] = delta / r
    return weights

@njit(cache=True)
def apply_transformation_numba(src, s, R, t):
    transformed = np.empty_like(src)
    for i in range(src.shape[0]):
        p = src[i]
        transformed[i] = s * (R @ p) + t
    return transformed

def robust_weighted_estimate_sim3_numba(src, tgt, init_weights, delta=0.1, max_iters=20, tol=1e-9, using_sim3 = True):
    src = src.astype(np.float32)
    tgt = tgt.astype(np.float32)
    init_weights = init_weights.astype(np.float32)

    s, R, t = weighted_estimate_sim3_numba(src, tgt, init_weights, using_sim3 = using_sim3)

    prev_error = float('inf')

    for iter in range(max_iters):
        transformed = apply_transformation_numba(src, s, R, t)
        residuals = compute_residuals_numba(tgt, transformed)

        print(f'Residuals: {np.mean(residuals)}')

        huber_weights = compute_huber_weights_numba(residuals, delta)
        combined_weights = init_weights * huber_weights
        combined_weights /= (np.sum(combined_weights) + 1e-12)

        s_new, R_new, t_new = weighted_estimate_sim3_numba(src, tgt, combined_weights, using_sim3 = using_sim3)

        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1)/2)))

        current_error = np.sum(huber_loss_numba(residuals, delta) * init_weights)

        if (param_change < tol and rot_angle < np.radians(0.1)) or \
           (abs(prev_error - current_error) < tol * prev_error):
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error

    return s, R, t

def warmup_numba():

    print("\nWarming up Numba JIT-compiled functions...")

    src = np.random.randn(50000, 3).astype(np.float32)
    tgt = np.random.randn(50000, 3).astype(np.float32)
    weights = np.ones(50000, dtype=np.float32)
    residuals = np.abs(np.random.randn(50000).astype(np.float32))
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    s = np.float32(1.0)
    delta = np.float32(1.0)

    try:
        _ = _weighted_estimate_sim3_numba(src, tgt, weights)
        print(" - _weighted_estimate_sim3_numba warmed up.")
    except Exception as e:
        print(" ! Failed to warm up _weighted_estimate_sim3_numba:", e)

    try:
        _ = _weighted_estimate_se3_numba(src, tgt, weights)
        print(" - _weighted_estimate_se3_numba warmed up.")
    except Exception as e:
        print(" ! Failed to warm up _weighted_estimate_se3_numba:", e)

    try:
        _ = huber_loss_numba(residuals, delta)
        print(" - huber_loss_numba warmed up.")
    except Exception as e:
        print(" ! Failed to warm up huber_loss_numba:", e)

    try:
        _ = compute_huber_weights_numba(residuals, delta)
        print(" - compute_huber_weights_numba warmed up.")
    except Exception as e:
        print(" ! Failed to warm up compute_huber_weights_numba:", e)

    try:
        _ = compute_residuals_numba(tgt, src)
        print(" - compute_residuals_numba warmed up.")
    except Exception as e:
        print(" ! Failed to warm up compute_residuals_numba:", e)

    try:
        _ = apply_transformation_numba(src, s, R, t)
        print(" - apply_transformation_numba warmed up.")
    except Exception as e:
        print(" ! Failed to warm up apply_transformation_numba:", e)

    print("Numba warm-up complete.\n")

# ===== Speed Up End =====




def weighted_align_point_maps(point_map1, conf1, point_map2, conf2, conf_threshold, config):
    """ point_map2 -> point_map1"""
    b1, _, _, _ = point_map1.shape
    b2, _, _, _ = point_map2.shape
    b = min(b1, b2)
    
    aligned_points1 = []
    aligned_points2 = []
    confidence_weights = []

    for i in range(b):
        mask1 = conf1[i] > conf_threshold
        mask2 = conf2[i] > conf_threshold
        valid_mask = mask1 & mask2

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        pts1 = point_map1[i][idx]
        pts2 = point_map2[i][idx]

        combined_conf = np.sqrt(conf1[i][idx] * conf2[i][idx])
        
        aligned_points1.append(pts1)
        aligned_points2.append(pts2)
        confidence_weights.append(combined_conf)

    if len(aligned_points1) == 0:
        raise ValueError("No matching point pairs were found!")

    all_pts1 = np.concatenate(aligned_points1, axis=0)
    all_pts2 = np.concatenate(aligned_points2, axis=0)
    all_weights = np.concatenate(confidence_weights, axis=0)

    print(f"The number of corresponding points matched: {all_pts1.shape[0]}")
    
    if config['Model']['align_method'] == 'numba':
        s, R, t = robust_weighted_estimate_sim3_numba(all_pts2, 
                                                all_pts1, 
                                                all_weights,
                                                delta=config['Model']['IRLS']['delta'],
                                                max_iters=config['Model']['IRLS']['max_iters'],
                                                tol=eval(config['Model']['IRLS']['tol']),
                                                using_sim3=config['Model']['using_sim3']
                                                )
    else: # numpy
        s, R, t = robust_weighted_estimate_sim3(all_pts2, 
                                                all_pts1, 
                                                all_weights,
                                                delta=config['Model']['IRLS']['delta'],
                                                max_iters=config['Model']['IRLS']['max_iters'],
                                                tol=eval(config['Model']['IRLS']['tol']),
                                                using_sim3=config['Model']['using_sim3']
                                                )

    mean_error = compute_alignment_error(
        point_map1, conf1, 
        point_map2, conf2, 
        conf_threshold, 
        s, R, t
    )
    print(f'Mean error: {mean_error}')

    return s, R, t

