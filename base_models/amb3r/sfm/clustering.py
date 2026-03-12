# References:
#   https://github.com/HengyiWang/amb3r/blob/main/sfm/clustering.py

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def farthest_point_sampling(dist, N=None, dist_thresh=None):
    """Farthest point sampling.

    Args:
        dist: NxN distance matrix.
        N: Number of points to sample.
        dist_thresh: Distance threshold. Point sampling terminates once the
                     maximum distance is below this threshold.

    Returns:
        indices: Indices of the sampled points.
    """
    if isinstance(dist, torch.Tensor):
        dist = dist.cpu().numpy()

    assert N is not None or dist_thresh is not None, "Either N or min_dist must be provided."

    if N is None:
        N = dist.shape[0]

    indices = []
    distances = [0]
    indices.append(np.random.choice(dist.shape[0]))
    for i in range(1, N):
        d = dist[indices].min(axis=0)
        bst = d.argmax()
        bst_dist = d[bst]
        if dist_thresh is not None and bst_dist < dist_thresh:
            break
        indices.append(bst)
        distances.append(bst_dist)
    return np.array(indices), np.array(distances)



def get_similarity_matrix(descriptors, whitening=False):
    """
    Compute a similarity matrix between feature descriptors.
    
    Args:
        descriptors (torch.Tensor): (num_images, C) feature matrix.
        whitening (bool): Whether to apply whitening before computing similarity.
        
    Returns:
        torch.Tensor: (num_images, num_images) similarity matrix.
    """
    with torch.cuda.amp.autocast(enabled=False):
        if whitening:
            # Step 1: Center
            mean = descriptors.mean(dim=0, keepdim=True)                     # (1, C)
            centered = descriptors - mean                                    # (num_images, C)

            # Step 2: Covariance and whitening
            cov = centered.T @ centered / (centered.size(0) - 1)             # (C, C)
            eps = 1e-6
            eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float32))  # (C,), (C, C)
            whitening_matrix = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + eps)) @ eigvecs.T
            descriptors = centered @ whitening_matrix                        # (num_images, C)
        else:
            # descriptors = descriptors - descriptors.mean(dim=0, keepdim=True)
            pass

        # Step 3: Normalize and compute similarity
        descriptors = F.normalize(descriptors, dim=1)
        similarity_matrix = descriptors @ descriptors.T                      # (num_images, num_images)
    
    similarity_matrix = (similarity_matrix + 1.0) / 2.0
    return similarity_matrix



def get_distance_matrix(descriptors, whitening=False):
    """
    Compute a distance matrix between feature descriptors.
    
    Args:
        descriptors (torch.Tensor): (num_images, C) feature matrix.
        whitening (bool): Whether to apply whitening before computing distance.
        
    Returns:
        torch.Tensor: (num_images, num_images) distance matrix.
    """
    similarity_matrix = get_similarity_matrix(descriptors, whitening=whitening)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix


def is_add_valid_by_dist(cluster_members, new_member_idx, dist_matrix, max_dist):
    """
    Checks if a new member is within max_dist of ALL existing members.
    """
    
    if not cluster_members: # If the set is empty
        return True
    
    # Get distances from the new member to all existing members
    dists = dist_matrix[new_member_idx, list(cluster_members)]
    
    # Check if any distance is greater than the max
    if torch.any(dists > max_dist):
        return False
    return True


def image_clustering(distance_matrix, min_frames_per_cluster, 
                     max_frames_per_cluster, refinement_iterations=2):
    """
    Forms clusters with min/max size constraints, determining N automatically.
    Uses mean inter-cluster similarity for merging, dynamic medoid updates,
    and an optional refinement phase.
    
    Args:
        distance_matrix (torch.Tensor): (N, N) distance matrix.
        min_frames_per_cluster (int): The minimum number of frames each cluster must have.
        max_frames_per_cluster (int): The maximum number of frames each cluster can have.
        refinement_iterations (int): Number of refinement passes after merging (default: 2).
        
    Returns:
        dict: {keyframe_idx: [list_of_member_indices_excluding_keyframe]}
    """
    
    num_total_frames = distance_matrix.shape[0]
    if isinstance(distance_matrix, np.ndarray):
        distance_matrix = torch.from_numpy(distance_matrix).float()
    
    similarity_matrix = 1.0 - distance_matrix

    # --- Helpers ---
    def get_medoid(members_set):
        """Find the most central member (medoid) of a cluster."""
        members_list = list(members_set)
        if len(members_list) <= 1:
            return members_list[0]
        dists = distance_matrix[members_list][:, members_list]
        return members_list[torch.argmin(dists.mean(dim=1)).item()]

    def mean_sim(c1, c2):
        """Mean pairwise similarity between two sets of frames."""
        if not c1 or not c2:
            return -float('inf')
        return similarity_matrix[list(c1)][:, list(c2)].mean().item()

    # --- Phase 1: Initial Over-Clustering ---
    initial_N = int(np.ceil(num_total_frames / min_frames_per_cluster))
    keyframes, _ = farthest_point_sampling(distance_matrix, N=initial_N) 
    
    # Assign all frames to their closest keyframe (guarantees full coverage)
    sim_to_keyframes = similarity_matrix[:, keyframes]
    cluster_assignments = torch.argmax(sim_to_keyframes, dim=1)
    
    clusters = {int(k): set() for k in keyframes}
    for img_idx, kf_local_idx in enumerate(cluster_assignments):
        global_kf_idx = keyframes[kf_local_idx]
        clusters[int(global_kf_idx)].add(img_idx)


    # --- Phase 2: Iterative Merging ---
    while True:
        # 1. Find the smallest cluster
        smallest_kf, smallest_size = -1, float('inf')
        for kf_idx, members in clusters.items():
            if len(members) < smallest_size:
                smallest_size = len(members)
                smallest_kf = kf_idx
        
        # 2. Check for termination
        if smallest_size >= min_frames_per_cluster:
            break 
        if len(clusters) <= 1:
            break

        # 3. Find the best merge target using mean inter-cluster similarity
        small_cluster_members = clusters[smallest_kf]
        best_target_idx = -1
        best_merge_sim = -float('inf')

        for target_kf_idx, target_members in clusters.items():
            if target_kf_idx == smallest_kf:
                continue
            # Only consider targets where merged size is within limits
            if len(target_members) + smallest_size <= max_frames_per_cluster:
                sim = mean_sim(small_cluster_members, target_members)
                if sim > best_merge_sim:
                    best_merge_sim = sim
                    best_target_idx = target_kf_idx

        # 4. Perform the merge or dissolve
        if best_target_idx != -1:
            # Merge is allowed — combine and update medoid
            merged = clusters[best_target_idx] | small_cluster_members
            del clusters[smallest_kf]
            del clusters[best_target_idx]
            new_medoid = get_medoid(merged)
            clusters[new_medoid] = merged
        else:
            # Cannot merge whole cluster — dissolve and reassign one-by-one
            del clusters[smallest_kf]
            for member_idx in small_cluster_members:
                best_home = -1
                best_sim = -float('inf')
                for kf_idx, members in clusters.items():
                    if len(members) < max_frames_per_cluster:
                        sim = similarity_matrix[member_idx, list(members)].mean().item()
                        if sim > best_sim:
                            best_sim = sim
                            best_home = kf_idx
                if best_home != -1:
                    clusters[best_home].add(member_idx)
                else:
                    # All clusters full — force-add to the closest one
                    kf_list = list(clusters.keys())
                    closest = kf_list[torch.argmax(similarity_matrix[member_idx, kf_list]).item()]
                    clusters[closest].add(member_idx)


    # --- Phase 3: Refinement ---
    for it in range(refinement_iterations):
        moved = 0
        current_kfs = list(clusters.keys())
        
        for kf in current_kfs:
            if kf not in clusters:
                continue
            for f in list(clusters[kf]):
                # Compute similarity of f to its current cluster (excluding itself)
                current_members_without_f = clusters[kf] - {f}
                if not current_members_without_f:
                    continue  # Don't move the last member
                current_sim = similarity_matrix[f, list(current_members_without_f)].mean().item()
                
                best_target, best_sim = kf, current_sim
                for target_kf in current_kfs:
                    if target_kf == kf or target_kf not in clusters:
                        continue
                    target_members = clusters[target_kf]
                    if len(target_members) >= max_frames_per_cluster:
                        continue
                    # Only move if source cluster won't go below min_size
                    if len(clusters[kf]) <= min_frames_per_cluster:
                        continue
                    sim = similarity_matrix[f, list(target_members)].mean().item()
                    if sim > best_sim:
                        best_target, best_sim = target_kf, sim
                
                if best_target != kf:
                    clusters[kf].remove(f)
                    clusters[best_target].add(f)
                    moved += 1
        
        # Update medoids after each refinement pass
        updated = {}
        for _, members in clusters.items():
            if len(members) > 0:
                new_medoid = get_medoid(members)
                updated[new_medoid] = members
        clusters = updated
        
        if moved == 0:
            break

    # --- Finalization ---
    # Exclude the keyframe from the members list.
    # Downstream code constructs views as [kf_idx] + members.
    final_clusters = {
        kf: sorted(members - {kf})
        for kf, members in clusters.items()
    }
    return final_clusters


def kf_clustering(
        distance_matrix,
        min_frames_per_cluster,
        max_frames_per_cluster,
        max_pose_distance,
        refinement_iterations=3
    ):
        """
        Hybrid constrained clustering with adaptive medoids, mean-similarity merging,
        and distance constraint preservation.
        """

        # --- Phase 0: Preprocessing ---
        if isinstance(distance_matrix, np.ndarray):
            distance_matrix = torch.from_numpy(distance_matrix).float()
        device = torch.device("cpu")
        distance_matrix = distance_matrix.to(device)
        similarity_matrix = (1.0 - distance_matrix).to(device)
        num_total_frames = distance_matrix.shape[0]

        # --- Phase 1: Initialization (same as v3) ---
        initial_N = int(np.ceil(num_total_frames / min_frames_per_cluster))
        keyframes, _ = farthest_point_sampling(distance_matrix, N=initial_N)

        clusters = {int(k): {int(k)} for k in keyframes}
        all_frames = set(range(num_total_frames))
        remaining_frames = list(all_frames - set(keyframes))
        unassigned_frames = set()

        for frame_idx in remaining_frames:
            best_cluster_idx, best_sim = -1, -float('inf')
            for kf_idx, members in clusters.items():
                is_size_valid = len(members) < max_frames_per_cluster
                is_dist_valid = is_add_valid_by_dist(members, frame_idx, distance_matrix, max_pose_distance)
                if is_size_valid and is_dist_valid:
                    sim = similarity_matrix[frame_idx, kf_idx]
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster_idx = kf_idx
            if best_cluster_idx != -1:
                clusters[best_cluster_idx].add(frame_idx)
            else:
                unassigned_frames.add(frame_idx)

        # --- Phase 2: Iterative merging with mean-similarity ---
        def cluster_medoid_static(c):
            c_list = list(c)
            if len(c_list) == 1:
                return c_list[0]
            dists = distance_matrix[c_list][:, c_list]
            mean_dists = dists.mean(dim=1)
            return int(c_list[torch.argmin(mean_dists)])

        def compute_mean_similarity(c1, c2):
            if not c1 or not c2:
                return -float('inf')
            return similarity_matrix[list(c1)][:, list(c2)].mean().item()

        island_clusters = set()
        while True:
            smallest_kf, smallest_members = None, None
            for kf, members in clusters.items():
                if kf not in island_clusters and (smallest_members is None or len(members) < len(smallest_members)):
                    smallest_kf, smallest_members = kf, members

            if smallest_members is None or len(smallest_members) >= min_frames_per_cluster:
                break

            best_target, best_sim = None, -float('inf')
            for kf_target, members_target in clusters.items():
                if kf_target == smallest_kf:
                    continue
                # Size constraint
                if len(members_target) + len(smallest_members) > max_frames_per_cluster:
                    continue
                # Distance constraint
                is_valid_merge = all(
                    is_add_valid_by_dist(members_target, m, distance_matrix, max_pose_distance)
                    for m in smallest_members
                )
                if not is_valid_merge:
                    continue

                sim = compute_mean_similarity(smallest_members, members_target)
                if sim > best_sim:
                    best_sim, best_target = sim, kf_target

            if best_target is not None:
                # Merge clusters
                merged_members = clusters[best_target] | smallest_members
                del clusters[smallest_kf]
                del clusters[best_target]
                new_medoid = cluster_medoid_static(merged_members)
                clusters[new_medoid] = merged_members
            else:
                island_clusters.add(smallest_kf)

        # --- Phase 3: Refinement ---
        for it in range(refinement_iterations):
            moved = 0
            current_kfs = list(clusters.keys())

            for kf in current_kfs:
                if kf not in clusters:
                    continue
                for f in list(clusters[kf]):
                    # Compute similarity of f to its current cluster (excluding itself)
                    current_members_without_f = clusters[kf] - {f}
                    if not current_members_without_f:
                        continue  # Don't move the last member
                    current_sim = similarity_matrix[f, list(current_members_without_f)].mean().item()

                    best_target, best_sim = kf, current_sim
                    for target_kf in current_kfs:
                        if target_kf == kf or target_kf not in clusters:
                            continue
                        target_members = clusters[target_kf]
                        if len(target_members) >= max_frames_per_cluster:
                            continue
                        # Don't move if source cluster would drop below min_size
                        if len(clusters[kf]) <= min_frames_per_cluster:
                            continue
                        if not is_add_valid_by_dist(target_members, f, distance_matrix, max_pose_distance):
                            continue
                        sim = similarity_matrix[f, list(target_members)].mean().item()
                        if sim > best_sim:
                            best_target, best_sim = target_kf, sim

                    if best_target != kf:
                        clusters[kf].remove(f)
                        clusters[best_target].add(f)
                        moved += 1

            # Update medoids after each refinement pass
            updated = {}
            for _, members in clusters.items():
                if len(members) > 0:
                    new_medoid = cluster_medoid_static(members)
                    updated[new_medoid] = members
            clusters = updated

            if moved == 0:
                break

        # --- Phase 4: Finalization ---
        final_clusters = {
            cluster_medoid_static(members): list(members)
            for _, members in clusters.items() if len(members) > 0
        }

        return final_clusters, list(unassigned_frames)


def evaluate_clustering(clusters, distance_matrix, connectivity_weight=0.5):
    """Evaluate clustering quality by intra-cluster compactness and inter-cluster connectivity.
    
    Lower is better.
    - Intra-cluster: mean distance from members to their keyframe (compactness).
    - Inter-cluster: max nearest-neighbor distance between keyframes (connectivity).
      Penalizes the worst isolated cluster that would cause registration failure.
    """
    if not clusters or len(clusters) <= 1:
        return float('inf')

    keyframes = list(clusters.keys())

    # 1. Intra-cluster compactness
    total_intra_dist = 0.0
    for kf, members in clusters.items():
        if not members:
            continue
        avg_dist_to_kf = distance_matrix[kf, members].mean().item()
        total_intra_dist += avg_dist_to_kf
    mean_intra_dist = total_intra_dist / len(clusters)

    # 2. Inter-cluster connectivity: mean nearest-neighbor distance among keyframes
    total_nn_dist = 0.0
    for i, kf1 in enumerate(keyframes):
        min_dist_to_other = float('inf')
        for j, kf2 in enumerate(keyframes):
            if i != j:
                dist = distance_matrix[kf1, kf2].item()
                if dist < min_dist_to_other:
                    min_dist_to_other = dist
        if min_dist_to_other < float('inf'):
            total_nn_dist += min_dist_to_other

    mean_nn_dist = total_nn_dist / len(keyframes)

    score = mean_intra_dist + connectivity_weight * mean_nn_dist
    return score


def evaluate_kf_clustering(clusters, unassigned_frames, distance_matrix, 
                            isolation_penalty=0.5, connectivity_weight=0.0):
    """Evaluate kf_clustering quality by mean pose distance + isolation penalty + connectivity.

    Lower is better. Uses pose distance from members to their cluster medoid,
    plus a penalty proportional to the number of isolated frames (unassigned 
    or in single-member clusters), plus inter-cluster connectivity.

    Args:
        clusters: {medoid_idx: [list_of_member_indices]}
        unassigned_frames: list of frame indices not assigned to any cluster.
        distance_matrix: (N, N) pose distance matrix.
        isolation_penalty: Weight for the isolation penalty term (default: 0.5).
        connectivity_weight: Weight for inter-cluster connectivity term (default: 0.5).

    Returns:
        float: Clustering score (lower is better).
    """
    total_frames = distance_matrix.shape[0]
    
    # 1. Mean distance from members to medoid (intra-cluster compactness)
    total_dist = 0.0
    num_clusters_with_members = 0
    num_isolated = len(unassigned_frames)

    for kf, members in clusters.items():
        if len(members) <= 1:
            # Single-member clusters are effectively isolated
            num_isolated += 1
            continue
        # Distance from all members to the medoid
        avg_dist_to_kf = distance_matrix[kf, members].mean().item()
        total_dist += avg_dist_to_kf
        num_clusters_with_members += 1

    if num_clusters_with_members > 0:
        mean_dist = total_dist / num_clusters_with_members
    else:
        mean_dist = 0.0

    # 2. Isolation penalty: fraction of isolated frames
    isolation_ratio = num_isolated / total_frames if total_frames > 0 else 0.0

    # 3. Inter-cluster connectivity: mean nearest-neighbor distance among medoids
    medoids = list(clusters.keys())
    mean_nn_dist = 0.0
    if len(medoids) > 1:
        total_nn_dist = 0.0
        for i, m1 in enumerate(medoids):
            min_dist = float('inf')
            for j, m2 in enumerate(medoids):
                if i != j:
                    dist = distance_matrix[m1, m2].item()
                    if dist < min_dist:
                        min_dist = dist
            if min_dist < float('inf'):
                total_nn_dist += min_dist
        mean_nn_dist = total_nn_dist / len(medoids)

    score = mean_dist + isolation_penalty * isolation_ratio + connectivity_weight * mean_nn_dist

    return score


def find_best_image_clustering(
    distance_matrix, 
    min_frames_per_cluster, 
    max_frames_per_cluster, 
    num_runs=50,
    refinement_iterations=2,
    patience=20,
):
    """Run image_clustering multiple times and return the best result.
    
    Since farthest_point_sampling has a random initial seed, running multiple 
    times and selecting the best result by intra-cluster distance improves 
    clustering robustness.
    
    Args:
        distance_matrix: (N, N) distance matrix.
        min_frames_per_cluster: Minimum cluster size.
        max_frames_per_cluster: Maximum cluster size.
        num_runs: Number of clustering runs (default: 5).
        refinement_iterations: Refinement iterations per run (default: 2).
        patience: Number of runs with no improvement before early stopping (default: 10).
        
    Returns:
        dict: {keyframe_idx: [list_of_member_indices_excluding_keyframe]}
    """
    best_clusters = None
    best_score = float('inf')
    no_improvement_count = 0
    
    iterator = tqdm(range(num_runs), desc="Clustering optimization")
    for i in iterator:
        clusters = image_clustering(
            distance_matrix=distance_matrix,
            min_frames_per_cluster=min_frames_per_cluster,
            max_frames_per_cluster=max_frames_per_cluster,
            refinement_iterations=refinement_iterations,
        )
        score = evaluate_clustering(clusters, distance_matrix)
        
        if score < best_score or best_clusters is None:
            best_score = score
            best_clusters = clusters
            no_improvement_count = 0
            iterator.set_postfix({"best_score": f"{best_score:.4f}"})
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= patience:
            print(f"Early stopping at run {i} (no improvement for {patience} runs)")
            break
    
    print(f"Best clustering score: {best_score:.4f}")
    return best_clusters


def find_best_kf_clustering(
    distance_matrix,
    conf,
    min_frames_per_cluster,
    max_frames_per_cluster,
    max_pose_distance,
    num_runs=10,
    refinement_iterations=3,
    isolation_penalty=0.5,
):
    """Run kf_clustering multiple times and return the best result.

    Since farthest_point_sampling has a random initial seed, running multiple
    times and selecting the best result improves clustering robustness.

    Args:
        distance_matrix: (N, N) pose distance matrix.
        min_frames_per_cluster: Minimum cluster size.
        max_frames_per_cluster: Maximum cluster size.
        max_pose_distance: Maximum pose distance for cluster membership.
        num_runs: Number of clustering runs (default: 5).
        refinement_iterations: Refinement iterations per run (default: 3).
        isolation_penalty: Weight for penalizing isolated frames (default: 0.5).

    Returns:
        tuple: (best_clusters_dict, best_unassigned_list)
    """
    best_clusters = None
    best_unassigned = None
    best_score = float('inf')

    for i in range(num_runs):
        clusters, unassigned = kf_clustering(
            distance_matrix=distance_matrix,
            min_frames_per_cluster=min_frames_per_cluster,
            max_frames_per_cluster=max_frames_per_cluster,
            max_pose_distance=max_pose_distance,
            refinement_iterations=refinement_iterations,
        )
        score = evaluate_kf_clustering(clusters, unassigned, distance_matrix,
                                       isolation_penalty=isolation_penalty)
        if score < best_score or best_clusters is None:
            best_score = score
            best_clusters = clusters
            best_unassigned = unassigned

    # print(f"Best kf_clustering score: {best_score:.4f} (out of {num_runs} runs)")
    return best_clusters, best_unassigned