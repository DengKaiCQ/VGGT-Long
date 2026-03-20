# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/tools/pts_vis.py

import os
import sys
import torch
import numpy as np
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'thirdparty'))

from segformer.model import EncoderDecoder


def _suppress_runtime_warnings(func):
    """Decorator to suppress RuntimeWarning during function execution."""
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return func(*args, **kwargs)
    return wrapper


def _make_sliding_view_1d(arr, win, step, axis=-1):
    """
    Produce a sliding-window view along one axis of an array.
    Returns shape (..., n_windows, ..., win) with the window dim appended last.
    """
    axis = axis % arr.ndim
    assert arr.shape[axis] >= win, (
        f"window ({win}) exceeds axis length ({arr.shape[axis]})"
    )
    n_win = (arr.shape[axis] - win + 1) // step
    out_shape = (*arr.shape[:axis], n_win, *arr.shape[axis + 1:], win)
    out_strides = (
        *arr.strides[:axis],
        step * arr.strides[axis],
        *arr.strides[axis + 1:],
        arr.strides[axis],
    )
    return np.lib.stride_tricks.as_strided(arr, shape=out_shape, strides=out_strides)


def _make_sliding_view_2d(arr, win_size, step, axes=(-2, -1)):
    """Produce a 2-D sliding-window view over two axes."""
    if isinstance(win_size, int):
        win_size = (win_size, win_size)
    if isinstance(step, int):
        step = (step, step)
    axes = [a % arr.ndim for a in axes]
    out = arr
    for i, ax in enumerate(axes):
        out = _make_sliding_view_1d(out, win_size[i], step[i], ax)
    return out


def _local_max_1d(arr, ksize, step, pad, axis=-1):
    """Max-pool along a single axis with padding."""
    axis = axis % arr.ndim
    if pad > 0:
        fill = np.nan if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
        pad_shape = (*arr.shape[:axis], pad, *arr.shape[axis + 1:])
        padding_block = np.full(pad_shape, fill_value=fill, dtype=arr.dtype)
        arr = np.concatenate([padding_block, arr, padding_block], axis=axis)
    windowed = _make_sliding_view_1d(arr, ksize, step, axis)
    return np.nanmax(windowed, axis=-1)


def _local_max_2d(arr, ksize, step, pad, axes=(-2, -1)):
    """Max-pool over two spatial axes with padding."""
    if isinstance(ksize, int):
        ksize = (ksize, ksize)
    if isinstance(step, int):
        step = (step, step)
    if isinstance(pad, int):
        pad = (pad, pad)
    axes_list = [a % arr.ndim for a in axes]
    out = arr
    for i, ax in enumerate(axes_list):
        out = _local_max_1d(out, ksize[i], step[i], pad[i], ax)
    return out


def _vec3_angular_distance(v1, v2, eps=1e-12):
    """
    Angle (in radians) between pairs of 3-D vectors (numpy).
    v1, v2 : (..., 3)
    """
    cross_mag = np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1) + eps
    dot_val = (v1 * v2).sum(axis=-1)
    return np.arctan2(cross_mag, dot_val)



@_suppress_runtime_warnings
def compute_surface_normals(pts, validity=None, grazing_angle_limit=None):
    """
    Derive a per-pixel normal map from a point map using finite differences
    and cross products of the four cardinal neighbours.

    Args:
        pts (np.ndarray) : (H, W, 3) point cloud organized as an image.
        validity (np.ndarray | None) : (H, W) bool mask of trustworthy pixels.
        grazing_angle_limit (float | None) : if set, discard normals whose
            angle to the view direction exceeds this many degrees.

    Returns:
        normals (np.ndarray) : (H, W, 3) unit normals (zero where invalid).
        normal_mask (np.ndarray) : (H, W) bool — returned **only** when
            *validity* is not None.
    """
    h, w = pts.shape[-3:-1]
    has_mask = validity is not None

    if validity is None:
        validity = np.ones(pts.shape[:-1], dtype=bool)

    # Pad both the mask and the point map with a 1-pixel border
    padded_mask = np.zeros((h + 2, w + 2), dtype=bool)
    padded_mask[1:-1, 1:-1] = validity

    padded_pts = np.zeros((h + 2, w + 2, 3), dtype=pts.dtype)
    padded_pts[1:-1, 1:-1, :] = pts

    # Finite-difference vectors to the four neighbours
    diff_up    = padded_pts[:-2,  1:-1, :] - padded_pts[1:-1, 1:-1, :]
    diff_left  = padded_pts[1:-1, :-2,  :] - padded_pts[1:-1, 1:-1, :]
    diff_down  = padded_pts[2:,   1:-1, :] - padded_pts[1:-1, 1:-1, :]
    diff_right = padded_pts[1:-1, 2:,   :] - padded_pts[1:-1, 1:-1, :]

    # Four cross-product normals (one per quadrant)
    quad_normals = np.stack([
        np.cross(diff_up,    diff_left,  axis=-1),
        np.cross(diff_left,  diff_down,  axis=-1),
        np.cross(diff_down,  diff_right, axis=-1),
        np.cross(diff_right, diff_up,    axis=-1),
    ])  # (4, H, W, 3)
    quad_normals /= (np.linalg.norm(quad_normals, axis=-1, keepdims=True) + 1e-12)

    # Validity per quadrant: both participating neighbours must be valid
    quad_valid = (
        np.stack([
            padded_mask[:-2,  1:-1] & padded_mask[1:-1, :-2],
            padded_mask[1:-1, :-2]  & padded_mask[2:,   1:-1],
            padded_mask[2:,   1:-1] & padded_mask[1:-1, 2:],
            padded_mask[1:-1, 2:]   & padded_mask[:-2,  1:-1],
        ])
        & padded_mask[None, 1:-1, 1:-1]
    )  # (4, H, W)

    if grazing_angle_limit is not None:
        view_ang = _vec3_angular_distance(padded_pts[None, 1:-1, 1:-1, :], quad_normals)
        view_ang = np.minimum(view_ang, np.pi - view_ang)
        quad_valid = quad_valid & (view_ang < np.deg2rad(grazing_angle_limit))

    # Average valid quadrant normals
    normals = (quad_normals * quad_valid[..., None]).sum(axis=0)
    normals /= (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    if has_mask:
        ok = quad_valid.any(axis=0)
        normals = np.where(ok[..., None], normals, 0)
        return normals, ok
    return normals


@_suppress_runtime_warnings
def detect_depth_discontinuities(depth, atol=None, rtol=None, ksize=3, validity=None):
    """
    Flag pixels whose depth differs significantly from at least one
    neighbour within a (ksize x ksize) window.

    Args:
        depth (np.ndarray) : (..., H, W) depth values.
        atol (float | None) : absolute threshold on max neighbour difference.
        rtol (float | None) : relative threshold (difference / depth).
        ksize (int) : neighbourhood size (must be odd).
        validity (np.ndarray | None) : mask for valid depth pixels.

    Returns:
        edges (np.ndarray) : (..., H, W) bool mask — True at discontinuities.
    """
    half = ksize // 2
    if validity is None:
        local_range = (
            _local_max_2d(depth,  ksize, step=1, pad=half)
            + _local_max_2d(-depth, ksize, step=1, pad=half)
        )
    else:
        local_range = (
            _local_max_2d(np.where(validity, depth, -np.inf),  ksize, step=1, pad=half)
            + _local_max_2d(np.where(validity, -depth, -np.inf), ksize, step=1, pad=half)
        )

    edges = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edges |= local_range > atol
    if rtol is not None:
        edges |= local_range / depth > rtol
    return edges


@_suppress_runtime_warnings
def detect_normal_discontinuities(normals, angle_tol, ksize=3, validity=None):
    """
    Flag pixels whose surface normal deviates sharply from at least
    one neighbour.

    Args:
        normals (np.ndarray) : (..., H, W, 3) surface normals.
        angle_tol (float) : tolerance in degrees.
        ksize (int) : neighbourhood size (must be odd).
        validity (np.ndarray | None) : (..., H, W) bool mask.

    Returns:
        edges (np.ndarray) : (..., H, W) bool mask.
    """
    assert normals.ndim >= 3 and normals.shape[-1] == 3
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    half = ksize // 2

    # Build the padded sliding-window view of the normals
    pad_widths = (
        *( [(0, 0)] * (normals.ndim - 3) ),
        (half, half),
        (half, half),
        (0, 0),
    )
    normals_padded = np.pad(normals, pad_widths, mode="edge")
    normals_win = _make_sliding_view_2d(
        normals_padded, win_size=ksize, step=1, axes=(-3, -2)
    )  # (..., H, W, 3, ksize, ksize)

    # Compute angular deviation between each pixel's normal and its window
    cos_vals = (normals[..., None, None] * normals_win).sum(axis=-3)

    if validity is None:
        max_angle = np.arccos(np.clip(cos_vals, -1, 1)).max(axis=(-2, -1))
    else:
        mask_padded = np.pad(
            validity,
            (*( [(0, 0)] * (validity.ndim - 3) ), (half, half), (half, half)),
            mode="edge",
        )
        mask_win = _make_sliding_view_2d(mask_padded, win_size=ksize, step=1, axes=(-3, -2))
        max_angle = np.where(
            mask_win,
            np.arccos(np.clip(cos_vals, -1, 1)),
            0,
        ).max(axis=(-2, -1))

    # Dilate the angular deviation map
    max_angle = _local_max_2d(max_angle, ksize, step=1, pad=half)
    return max_angle > np.deg2rad(angle_tol)



def get_pts_edge_mask(pts, sky_mask=None, conf=None, conf_threshold=1e-2, edge_normal_threshold=5.0, edge_depth_threshold=0.008):
    """
    Build a per-pixel validity mask that removes points near depth or
    normal discontinuities.

    Args:
        pts (np.ndarray) : (B, H, W, 3) world-space point cloud.
        edge_normal_threshold (float) : angle tolerance in degrees.
        edge_depth_threshold (float) : relative depth tolerance.

    Returns:
        mask (np.ndarray) : (B, H, W) bool mask of clean points.
    """
    mask = np.ones(pts.shape[:-1], dtype=bool)

    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    if isinstance(sky_mask, torch.Tensor):
        sky_mask = sky_mask.cpu().numpy()
    if isinstance(conf, torch.Tensor):
        conf = conf.cpu().numpy()

    if sky_mask is not None:
        mask &= ~sky_mask
    
    if conf is not None:
        mask &= conf >= conf_threshold

    edge_masks = []
    for b in range(len(pts)):
        frame_mask = mask[b]  # (H, W)
        frame_pts  = pts[b]   # (H, W, 3)

        if frame_mask.any():
            # Surface normals
            normals, normals_ok = compute_surface_normals(
                frame_pts, validity=frame_mask
            )
            normal_edges = detect_normal_discontinuities(
                normals, angle_tol=edge_normal_threshold, validity=normals_ok
            )

            # Depth discontinuities (z-channel)
            z_vals = frame_pts[:, :, -1]
            depth_edges = detect_depth_discontinuities(
                z_vals, rtol=edge_depth_threshold, validity=frame_mask
            )

            # A pixel is kept only if it is NOT an edge in both senses
            combined_edge = depth_edges & normal_edges
            edge_masks.append(~combined_edge)
        else:
            edge_masks.append(np.zeros_like(frame_mask, dtype=bool))

    edge_mask = np.stack(edge_masks, axis=0)
    mask = mask & edge_mask
    return mask




def get_sky_mask(images, chunk_size=20, segformer_path='./checkpoints/segformer.b0.512x512.ade.160k.pth'):
    '''Generate a sky mask for a batch of images using a pre-trained SegFormer model.
    
    Args:
        - images (torch.Tensor): A batch of images with shape (B, T, C, H, W) in [0, 1] range.
        - chunk_size (int): Number of images to process in one chunk to manage GPU memory.
    Returns:
        - sky_mask (np.ndarray): A boolean mask of shape (T, H, W) where True indicates sky pixels.
    '''
    segformer = EncoderDecoder()
    segformer.load_state_dict(torch.load(segformer_path, map_location=torch.device('cpu'), weights_only=False)['state_dict'])
    segformer.cuda()
    segformer.eval()
    with torch.no_grad():
        # Get original dimensions
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, T, C, H, W = images.shape
        
        # Reshape for processing. Assuming B=1, shape becomes (T, C, H, W)
        images_reshaped = images.view(B * T, C, H, W)

        sky_mask_chunks = []
        # Process the images in chunks
        for i in range(0, T, chunk_size):
            # Get the current chunk from the tensor
            chunk = images_reshaped[i:i + chunk_size]
            
            # Run inference on the chunk
            seg_logits_chunk = segformer.inference_(chunk.cuda())
            
            # Create the mask for the chunk and move it to CPU to free GPU memory
            sky_mask_chunk = (seg_logits_chunk == 2).cpu()
            
            sky_mask_chunks.append(sky_mask_chunk)

        # Concatenate all the processed chunks into the final mask
        sky_mask = torch.cat(sky_mask_chunks, dim=0) # Final shape: (T, H, W)
        sky_mask = sky_mask.numpy()
    
    return sky_mask


def get_pts_mask(pts, images=None, conf=None, conf_threshold=1e-2, 
                 edge_normal_threshold=5.0, edge_depth_threshold=0.008,
                 segformer_path='./checkpoints/segformer.b0.512x512.ade.160k.pth'):
    '''
    Generate a mask for points based on edge detection and sky segmentation.
    '''
    sky_mask = None
    if images is not None:
        sky_mask = get_sky_mask(images, segformer_path=segformer_path)

    edge_mask = get_pts_edge_mask(pts, sky_mask=sky_mask, conf=conf, conf_threshold=conf_threshold, edge_normal_threshold=edge_normal_threshold, edge_depth_threshold=edge_depth_threshold)
    
    return edge_mask, sky_mask