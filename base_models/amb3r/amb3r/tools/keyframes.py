# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/tools/keyframes.py

import torch

def select_keyframes_iteratively(
    dists: torch.Tensor,
    conf: torch.Tensor,
    keyframe_threshold: float,
    keyframe_indices = [0],
    tolerance: float = 5e-3) -> torch.Tensor:
    """
    Selects keyframes iteratively based on distance and confidence.

    Builds upon an initial set of keyframes by repeatedly adding a new keyframe.
    The selection process is as follows:
    1. Identify all frames that are sufficiently distant from existing keyframes.
    2. From this set, find the frame with the maximum confidence score.
    3. Identify all other frames with a confidence score within `tolerance` of the maximum.
    4. From this final group, select the one with the largest frame index as the new keyframe.

    Args:
        dists (torch.Tensor): A (T, T) tensor of pairwise extrinsic distances.
        conf (torch.Tensor): A (T, H, W) tensor of world point confidence maps.
        keyframe_threshold (float): The minimum distance for selecting a new keyframe.
            (This is assumed to be an attribute of `self`).
        keyframe_indices (List[int]): An optional list of indices to use as the
            starting keyframes. Defaults to `[0]`.
        tolerance (float): The tolerance for confidence scores when selecting the
            best keyframe. Frames within this tolerance of the max confidence
            are considered candidates for the final selection.

    Returns:
        torch.Tensor: A (T,) boolean tensor where True indicates a selected keyframe.
    """
    num_frames = dists.shape[0]

    all_indices = set(range(num_frames))
    # Ensure keyframe_indices is a mutable copy for this run
    keyframe_indices = list(keyframe_indices)
    candidate_indices = all_indices - set(keyframe_indices)

    avg_confidence = conf.mean(dim=(-1, -2))

    while True:
        potential_next_keyframes = []
        for c_idx in candidate_indices:
            is_far_enough = all(dists[c_idx, k_idx] > keyframe_threshold for k_idx in keyframe_indices)
            if is_far_enough:
                potential_next_keyframes.append(c_idx)

        if not potential_next_keyframes:
            break

        max_conf = max(avg_confidence[i] for i in potential_next_keyframes)

        next_keyframe_idx = min(
            i for i in potential_next_keyframes if avg_confidence[i] >= max_conf * (1 - tolerance)
        )

        keyframe_indices.append(next_keyframe_idx)
        candidate_indices.remove(next_keyframe_idx)

    is_keyframe = torch.zeros(num_frames, dtype=torch.bool, device=dists.device)
    is_keyframe[torch.tensor(keyframe_indices, dtype=torch.long)] = True
    
    return is_keyframe

