# References:
#   https://github.com/HengyiWang/amb3r/blob/main/sfm/pipeline.py

import torch
import heapq
from tqdm import tqdm
from omegaconf import OmegaConf
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from sfm.clustering import image_clustering, get_distance_matrix, find_best_image_clustering
from sfm.memory import SfMemory



class AMB3R_SfM():
    def __init__(self, model, cfg_path='./sfm/sfm_config.yaml'):
        self.cfg = OmegaConf.load(cfg_path)
        self.model = model.to(self.cfg.device)


    def extract_features(self, images, chunk_size=128):
        '''
        Feature extraction for clustering all images
        '''

        if images.shape[0] == 1:
            images = images.squeeze(0)
        T, C, H, W = images.shape

        all_descriptors = []


        for i in range(0, T, chunk_size):
            images_chunk = images[i : i + chunk_size] # Shape: (chunk, C, H, W)
            views = {
                'images': images_chunk.unsqueeze(0).to(self.model.device)
                }  # Add batch dim: (1, chunk, C, H, W)

            feat_chunk = self.model.extract_amb3r_sfm_features(views).cpu()  # patch_tokens['x_norm_patchtokens'] shape: (chunk, N, C)
            all_descriptors.append(feat_chunk)
            
        feature_descriptors = torch.cat(all_descriptors, dim=0) # Shape: (T, C)

        return feature_descriptors
    

    @torch.no_grad()
    def local_mapping(self, views_all, cfg, keyframe_memory=None, benchmark_conf0=None):
        '''
        Local mapping for one cluster
        '''

        with torch.autocast(device_type=self.model.device, dtype=torch.bfloat16):
            res = self.model.run_amb3r_sfm(views_all, cfg, keyframe_memory, benchmark_conf0=benchmark_conf0)
        return res
    
    
    def process_one_cluster(self, images, cluster_kf_idx, cluster_member_indices):
        '''
        Run inference on one cluster.
        '''

        if images.shape[0] != 1:
            images = images.unsqueeze(0) # 1, T, C, H, W

        views_to_map = {
            'images': images[:, [cluster_kf_idx] + cluster_member_indices].to(self.model.device),
        }

        res = self.local_mapping(views_to_map, self.cfg)

        cluster_prediction = {
            'pts': res['world_points'][0].cpu(), # T, H, W, 3
            'conf': res['world_points_conf'][0].cpu(), # T, H, W
            'pose': res['pose'][0].cpu(), # T, 4, 4
        }
        cluster_prediction['conf_sig'] = (cluster_prediction['conf'] - 1) / cluster_prediction['conf']
        cluster_prediction['conf_mean'] = cluster_prediction['conf_sig'].mean()
        cluster_prediction['conf_sig_mean'] = cluster_prediction['conf_sig'].mean()

        return cluster_prediction


    def compute_cluster_affinity(self, source_indices, target_groups, pairwise_matrix, k, alpha=0.3, higher_is_better=True):
        """
        Computes the mean pairwise score between source_indices and each target group.
        """

        group_scores = []
        for group in target_groups:
            sub_matrix = pairwise_matrix[source_indices, :][:, group]
            if sub_matrix.numel() > 0:
                group_scores.append(sub_matrix.mean().item())

        if not group_scores:
            return None

        all_mean = sum(group_scores) / len(group_scores)
        group_scores.sort(reverse=higher_is_better)
        top_k = group_scores[:min(k, len(group_scores))]
        top_k_mean = sum(top_k) / len(top_k)

        return alpha * top_k_mean + (1.0 - alpha) * all_mean


    def build_prediction_dict(self, res, idx_to_map, **extra_fields):
        '''
        Build a standard prediction dict from local_mapping results.
        '''

        conf = res['world_points_conf'][0].cpu()
        prediction = {
            'pts': res['world_points'][0].cpu(),
            'conf': conf,
            'pose': res['pose'][0].cpu(),
            'conf_sig': (conf - 1) / conf,
            'idx': idx_to_map,
            'iter': torch.ones(len(idx_to_map)),
        }
        prediction.update(extra_fields)
        return prediction


    def find_low_confidence_frames(self, clusters, cluster_predictions):
        """
        Identify frames with mean confidence below threshold across all clusters.
        """

        frames_to_isolate = set()
        for cluster_kf_idx, cluster_member_indices in clusters.items():
            cluster_pred = cluster_predictions[cluster_kf_idx]
            all_indices = [cluster_kf_idx] + cluster_member_indices
            mean_confs_per_frame = cluster_pred['conf_sig'].mean(dim=(1, 2))
            low_conf_mask = mean_confs_per_frame < self.cfg.cluster_frame_conf_threshold
            low_conf_indices = [idx for i, idx in enumerate(all_indices) if low_conf_mask[i]]
            frames_to_isolate.update(low_conf_indices)
        return frames_to_isolate


    def restructure_clusters(self, clusters, cluster_predictions, frames_to_isolate):
        """
        Restructure clusters by isolating low-confidence frames and promoting new keyframes.
        """

        modified_clusters = {}

        for kf_idx, members in clusters.items():
            high_conf_members = [m for m in members if m not in frames_to_isolate]
            low_conf_members = [m for m in members if m in frames_to_isolate]

            # Isolate each low-conf member as a single-frame cluster
            for iso_member in low_conf_members:
                if iso_member not in modified_clusters:
                    modified_clusters[iso_member] = []

            if kf_idx in frames_to_isolate:
                # Keyframe itself is low-conf: isolate it
                if kf_idx not in modified_clusters:
                    modified_clusters[kf_idx] = []

                if high_conf_members:
                    # Promote the most confident member as the new keyframe
                    cluster_pred = cluster_predictions[kf_idx]
                    all_indices = [kf_idx] + members
                    conf_per_frame = cluster_pred['conf_sig'].mean(dim=(1, 2))
                    idx_conf = {idx: conf_per_frame[i].item() for i, idx in enumerate(all_indices) if idx in high_conf_members}
                    new_kf = max(idx_conf, key=idx_conf.get)
                    new_members = [m for m in high_conf_members if m != new_kf]
                    modified_clusters[new_kf] = new_members
            else:
                # Keyframe is high-conf: keep it with its high-conf members
                modified_clusters[kf_idx] = high_conf_members

        for iso_frame in frames_to_isolate:
            if iso_frame not in modified_clusters:
                modified_clusters[iso_frame] = []

        return modified_clusters


    def process_clusters(self, images, clusters, distance_matrix=None):
        '''
        Run inference on each cluster and select the best cluster to initialize the map.
        '''

        cluster_predictions = {}
        best_cluster_kf_idx = -1
        max_combined_score = -float('inf')

        if distance_matrix is not None:
            similarity_matrix = 1.0 - distance_matrix
        else:
            similarity_matrix = None

        for cluster_kf_idx, cluster_member_indices in tqdm(clusters.items(), desc="Processing clusters"):
            cluster_pred = self.process_one_cluster(images, cluster_kf_idx, cluster_member_indices)
            cluster_predictions[cluster_kf_idx] = cluster_pred
            
            conf_mean = cluster_pred['conf_mean'].item()

            if similarity_matrix is not None and len(clusters) > 1:
                all_frames = [cluster_kf_idx] + cluster_member_indices
                target_groups = [
                    [other_kf] + other_members
                    for other_kf, other_members in clusters.items()
                    if other_kf != cluster_kf_idx
                ]
                result = self.compute_cluster_affinity(
                    all_frames, target_groups, similarity_matrix,
                    k=self.cfg.k_kf_candidates_clusters,
                )
                similarity = result if result is not None else 0.0
            else:
                similarity = 1.0 

            combined_score = conf_mean * similarity

            if combined_score > max_combined_score:
                max_combined_score = combined_score
                best_cluster_kf_idx = cluster_kf_idx
                if similarity_matrix is not None and len(clusters) > 1:
                    print(f"  [Init Seed] New best {cluster_kf_idx}: Conf={conf_mean:.4f}, Sim={similarity:.4f}, Score={combined_score:.4f}")

        print(f"Selected Initial Keyframe: {best_cluster_kf_idx} with Combined Score {max_combined_score:.4f}")

        # Identify and isolate low-confidence frames, then restructure clusters
        frames_to_isolate = self.find_low_confidence_frames(clusters, cluster_predictions)
        print(f"Isolating {len(frames_to_isolate)} low-confidence frames into single-frame clusters.")
        modified_clusters = self.restructure_clusters(clusters, cluster_predictions, frames_to_isolate)
        
        return modified_clusters, best_cluster_kf_idx


    def initialize_map(self, images, cluster_kf_idx, cluster_member_indices):
        '''
        Careful map initialization 
        '''

        best_kf_idx = cluster_kf_idx

        views_to_map = {
                'images': images[:, [cluster_kf_idx] + cluster_member_indices].to(self.model.device),
            }

        res = self.local_mapping(views_to_map, self.cfg)

        # Initialization
        initial_cluster_pred = self.build_prediction_dict(res, [cluster_kf_idx] + cluster_member_indices)
        best_kf_conf = initial_cluster_pred['conf'][0].mean()
        print(f"Initial keyframe idx: {cluster_kf_idx} with mean conf {best_kf_conf}")

        cluster_pred_all = {}
        cluster_pred_all[best_kf_idx] = {
            'pred': initial_cluster_pred,
            'idx': [cluster_kf_idx] + cluster_member_indices
        }

        # Find the best first keyframe to start with
        for member_idx in cluster_member_indices:

            idx_to_use = [member_idx] + [i for i in cluster_member_indices if i != member_idx] + [cluster_kf_idx]
            views_to_map = {
                'images': images[:, idx_to_use].to(self.model.device),
            }
            res = self.model.run_amb3r_sfm(views_to_map, self.cfg)
            cluster_pred = self.build_prediction_dict(res, idx_to_use)

            cluster_pred_all[member_idx] = {
                'pred': cluster_pred,
                'idx': idx_to_use
            }

            if cluster_pred['conf'][0].mean() > best_kf_conf:
                best_kf_conf = cluster_pred['conf'][0].mean()
                best_kf_idx = member_idx
        
        print(f"Refined best keyframe idx: {best_kf_idx} with mean conf {best_kf_conf}")

        self.keyframe_memory.initialize(cluster_pred_all, best_kf_idx)


    def rank_clusters_by_distance(self, rest_clusters_kf_indices, clusters, distance_matrix,
                                  processed_clusters, total_clusters):
        '''
        Rank remaining clusters by similarity to existing global KF clusters.
        '''

        cluster_distances = {}
        for kf_idx in rest_clusters_kf_indices:
            member_indices = clusters[kf_idx]
            all_frames_in_cluster = [kf_idx] + member_indices
            
            target_groups = [cluster for cluster in self.keyframe_memory.kf_clusters if cluster]

            result = self.compute_cluster_affinity(
                all_frames_in_cluster, target_groups, distance_matrix,
                k=self.cfg.k_kf_candidates_clusters,
                higher_is_better=False,
            )
            cluster_distances[kf_idx] = result if result is not None else float('inf')

        sorted_clusters = sorted(cluster_distances.items(), key=lambda item: item[1])
        # Use 2x candidates for first 10% of clusters for stable init
        k_mult = 2 if processed_clusters < total_clusters * 0.1 else 1
        top_k_candidates = sorted_clusters[:self.cfg.k_candidate_clusters * k_mult]
        return top_k_candidates, k_mult


    def register_candidate_cluster(self, images, distance_matrix, clusters,
                                   rest_clusters_kf_indices, candidate_kf_idx,
                                   candidate_results, k_mult):
        '''Register one candidate cluster against top-k global KF clusters.
        
        Mutates clusters and rest_clusters_kf_indices in-place if merged non-KFs
        require keyframe promotion. Populates candidate_results[candidate_kf_idx].
        
        Returns:
            candidate_kf_idx: may differ from input if keyframe was promoted
            should_early_stop: True if confidence exceeds early-stop threshold
        '''
        benchmark_conf0 = None
        candidate_frames = [candidate_kf_idx] + clusters[candidate_kf_idx]
        if not candidate_frames:
            return candidate_kf_idx, False

        global_cluster_distances = []
        for global_kf_cluster in self.keyframe_memory.kf_clusters:
            if not global_kf_cluster:
                global_cluster_distances.append((float('inf'), global_kf_cluster))
                continue
            
            dist_sub_matrix = distance_matrix[candidate_frames, :][:, global_kf_cluster]
            avg_dist = dist_sub_matrix.mean() if dist_sub_matrix.numel() > 0 else float('inf')
            global_cluster_distances.append((avg_dist, global_kf_cluster))
        
        sorted_global_clusters = sorted(global_cluster_distances, key=lambda item: item[0])
        top_k_global_clusters = sorted_global_clusters[:self.cfg.k_kf_candidates_clusters * k_mult]

        all_merged_non_kf = []  # Collect across all global cluster iterations

        for avg_dist, global_kf_cluster in top_k_global_clusters:

            idx_to_map = global_kf_cluster + [candidate_kf_idx] + clusters[candidate_kf_idx]
            views_to_map = {
                'images': images[:, idx_to_map].to(self.model.device),
                'kf_idx': global_kf_cluster,
            }

            benchmark_conf_all = torch.cat([10 * torch.ones(len(global_kf_cluster)), benchmark_conf0]) if benchmark_conf0 is not None else None

            res = self.local_mapping(views_to_map, self.cfg, keyframe_memory=self.keyframe_memory, benchmark_conf0=benchmark_conf_all)

            if res is None:
                continue

            aligned_res, merged_non_kf = self.keyframe_memory.update_kf(
                res, idx_to_map, num_kf=len(global_kf_cluster),
                non_kf_conf_threshold=self.cfg.conf_min_early_stop_threshold
            )

            if aligned_res is None:
                del res
                continue

            # Collect merged non-KF indices (don't remove yet — would break idx_to_map in next iteration)
            if merged_non_kf:
                all_merged_non_kf.extend(merged_non_kf)

            if candidate_kf_idx not in candidate_results:
                num_kf = len(global_kf_cluster)
                candidate_results[candidate_kf_idx] = {
                    'pts': aligned_res['pts'][num_kf:],
                    'pose': aligned_res['pose'][num_kf:],
                    'conf_sig': aligned_res['conf_sig'][num_kf:],
                    'iter': aligned_res['iter'][num_kf:],
                    'idx': aligned_res['idx'][num_kf:],
                }
                if 'benchmark_conf0' in res:
                    benchmark_conf0 = res['benchmark_conf0'][num_kf:].cpu().clone()
                else:
                    benchmark_conf0 = res['world_points_conf'][0].cpu().mean(dim=(1,2))[num_kf:].clone()
            else:
                num_kf = len(global_kf_cluster)
                # For each frame, only keep the best aligned prediction (based on mean confidence)
                candidate_conf = candidate_results[candidate_kf_idx]['conf_sig'].mean(dim=(1,2))
                new_conf = aligned_res['conf_sig'][num_kf:].mean(dim=(1,2))

                # If the new prediction is better, update the candidate results
                better_mask = new_conf > candidate_conf
                candidate_results[candidate_kf_idx]['pts'][better_mask] = aligned_res['pts'][num_kf:][better_mask]
                candidate_results[candidate_kf_idx]['conf_sig'][better_mask] = aligned_res['conf_sig'][num_kf:][better_mask]
                candidate_results[candidate_kf_idx]['pose'][better_mask] = aligned_res['pose'][num_kf:][better_mask]
                candidate_results[candidate_kf_idx]['iter'][better_mask] = aligned_res['iter'][num_kf:][better_mask]

                if 'benchmark_conf0' in res:
                    benchmark_conf0[better_mask] = res['benchmark_conf0'][num_kf:][better_mask].cpu()
                else:
                    benchmark_conf0[better_mask] = res['world_points_conf'][0].cpu().mean(dim=(1,2))[num_kf:][better_mask]
            

            del res

        # Remove merged non-KF frames from the candidate cluster after the inner loop
        if all_merged_non_kf:
            remaining_members = [m for m in clusters[candidate_kf_idx] if m not in all_merged_non_kf]

            if candidate_kf_idx in all_merged_non_kf:
                # Candidate keyframe itself was merged — promote a new keyframe
                orig_kf_idx = candidate_kf_idx
                del clusters[orig_kf_idx]
                if remaining_members:
                    # Pick the member closest to the original keyframe by visual distance
                    dists_to_orig = distance_matrix[orig_kf_idx, remaining_members]
                    new_kf_local = dists_to_orig.argmin().item()
                    new_kf_idx = remaining_members[new_kf_local]
                    new_members = [m for m in remaining_members if m != new_kf_idx]
                    clusters[new_kf_idx] = new_members
                    print(f"  [coarse_reg] Promoted new KF {new_kf_idx} (was {orig_kf_idx}) with {len(new_members)} members")
                    # Move candidate_results entry to the new key
                    if orig_kf_idx in candidate_results:
                        candidate_results[new_kf_idx] = candidate_results.pop(orig_kf_idx)
                    # Update rest_clusters_kf_indices
                    if orig_kf_idx in rest_clusters_kf_indices:
                        rest_clusters_kf_indices[rest_clusters_kf_indices.index(orig_kf_idx)] = new_kf_idx
                    candidate_kf_idx = new_kf_idx
            else:
                clusters[candidate_kf_idx] = remaining_members

        print('Min confidence for candidate cluster {}: {:.4f}'.format(
            candidate_kf_idx,
            candidate_results[candidate_kf_idx]['conf_sig'].mean(dim=(1,2)).min()
        ))

        # Early stopping if we already have a very good candidate
        should_early_stop = candidate_results[candidate_kf_idx]['conf_sig'].mean(dim=(1,2)).min() > self.cfg.conf_min_early_stop_threshold
        return candidate_kf_idx, should_early_stop


    def remap_unmapped_frames(self, images, distance_matrix, unmapped_frames):
        '''
        Re-map remaining low-confidence frames by trying each against existing KF clusters.
        '''
        if not unmapped_frames:
            return

        print(f"\nRe-mapping {len(unmapped_frames)} remaining low-confidence frames...")

        # Sort unmapped frames by their closest distance to any existing KF cluster
        frame_distances = []
        for frame_idx in unmapped_frames:
            min_dist = float('inf')
            for kf_cluster in self.keyframe_memory.kf_clusters:
                if not kf_cluster:
                    continue
                dist_sub_matrix = distance_matrix[[frame_idx], :][:, kf_cluster]
                if dist_sub_matrix.numel() > 0:
                    cluster_dist = dist_sub_matrix.mean().item()
                    min_dist = min(min_dist, cluster_dist)
            frame_distances.append((min_dist, frame_idx))
        
        frame_distances.sort(key=lambda x: x[0])
        print(f"Processing unmapped frames in distance order: {[(idx, f'{d:.4f}') for d, idx in frame_distances]}")

        for _, frame_idx in frame_distances:
            best_local_pred = None
            best_mean_conf = -float('inf')

            # Try mapping against each KF cluster
            for kf_cluster in self.keyframe_memory.kf_clusters:
                
                idx_to_map = kf_cluster + [frame_idx]
                views_to_map = {
                    'images': images[:, idx_to_map].to(self.model.device),
                    'kf_idx': kf_cluster,
                    }
                res = self.local_mapping(views_to_map, self.cfg, keyframe_memory=self.keyframe_memory)

                prediction_local = self.build_prediction_dict(res, idx_to_map, kf_cluster=kf_cluster)
                
                current_conf = prediction_local['conf_sig'].mean()
                if current_conf > best_mean_conf:
                    best_mean_conf = current_conf
                    best_local_pred = prediction_local
                
                del res
                
                # Early stop if the confidence is high enough
                if best_mean_conf > self.cfg.conf_min_early_stop_threshold:
                    break

            # After trying all clusters, fuse the best result if it's good enough
            if best_mean_conf > self.cfg.low_conf_frame_threshold:
                self.keyframe_memory.update(best_local_pred, align=True, 
                                            num_kf=len(best_local_pred['kf_cluster']),
                                            add_kf=True)
            else:
                if self.cfg.allow_unmapped_frames:
                    print(f"Skipping unmapped frame {frame_idx} (conf {best_mean_conf:.4f})")
                    self.keyframe_memory.add_unmapped_frames({frame_idx})
                else:
                    print(f"Failed to re-map frame {frame_idx}, max conf {best_mean_conf:.4f}")
                    self.keyframe_memory.update(best_local_pred)
        
        print("Low-confidence frame re-mapping complete.")


    def coarse_registration(self, images, distance_matrix, rest_clusters_kf_indices, clusters):
        '''
        Coarse registration of each unmapped cluster.
        '''

        unmapped_low_conf_frames = set()
        prev_unmapped_count = None
        round_num = 0
        total_clusters = len(rest_clusters_kf_indices)
        processed_clusters = 0

        while True:
            round_num += 1
            if round_num > self.cfg.max_coarse_registration_iters:
                break
            print(f"\n=== Coarse registration round {round_num} with {len(rest_clusters_kf_indices)} clusters ===")
            pbar = tqdm(total=len(rest_clusters_kf_indices), desc=f"Coarse Reg Round {round_num}")

            while len(rest_clusters_kf_indices) > 0:
                # Step 1: Rank remaining clusters by affinity to existing global KF clusters
                top_k_candidates, k_mult = self.rank_clusters_by_distance(
                    rest_clusters_kf_indices, clusters, distance_matrix,
                    processed_clusters, total_clusters)

                # Step 2: Register each candidate against its top-k global KF clusters
                candidate_results = {}
                best_candidate_kf_idx = -1
                best_candidate_conf_mean = -float('inf')

                for candidate_kf_idx, _ in top_k_candidates:
                    candidate_kf_idx, should_early_stop = self.register_candidate_cluster(
                        images, distance_matrix, clusters, rest_clusters_kf_indices,
                        candidate_kf_idx, candidate_results, k_mult)

                    if should_early_stop:
                        best_candidate_kf_idx = candidate_kf_idx
                        best_candidate_conf_mean = candidate_results[candidate_kf_idx]['conf_sig'].mean()
                        print("Early stopping: found a very good candidate cluster.")
                        break

                    conf_mean = candidate_results[candidate_kf_idx]['conf_sig'].mean()
                    if conf_mean > best_candidate_conf_mean:
                        best_candidate_conf_mean = conf_mean
                        best_candidate_kf_idx = candidate_kf_idx

                # Step 3: Filter and commit the best candidate
                print(f"Adding cluster with keyframe {best_candidate_kf_idx} and mean confidence {best_candidate_conf_mean:.4f}")
                best_candidate_result = candidate_results[best_candidate_kf_idx]

                mean_confs = best_candidate_result['conf_sig'].mean(dim=(1,2))
                low_conf_mask = mean_confs < self.cfg.low_conf_frame_threshold
                high_conf_mask = ~low_conf_mask
                
                # Get frames to defer
                low_conf_indices = [idx for i, idx in enumerate(best_candidate_result['idx']) if low_conf_mask[i]]
                if low_conf_indices:
                    print(f"Deferring {len(low_conf_indices)} low-conf frames: {low_conf_indices}")
                    unmapped_low_conf_frames.update(low_conf_indices)
                
                # Filter the best_candidate_result to only include high-conf frames
                best_candidate_result_filtered = {
                    'pts': best_candidate_result['pts'][high_conf_mask],
                    'conf_sig': best_candidate_result['conf_sig'][high_conf_mask],
                    'pose': best_candidate_result['pose'][high_conf_mask],
                    'iter': best_candidate_result['iter'][high_conf_mask],
                    'idx': [idx for i, idx in enumerate(best_candidate_result['idx']) if high_conf_mask[i]],
                    }
                
                if len(best_candidate_result_filtered['idx']) > 0:
                    self.keyframe_memory.update(best_candidate_result_filtered)
                
                rest_clusters_kf_indices.remove(best_candidate_kf_idx)
                processed_clusters += 1
                pbar.update(1)
                pbar.set_postfix(remaining=len(rest_clusters_kf_indices), conf=f"{best_candidate_conf_mean:.4f}")

            pbar.close()

            # After all clusters in this round are processed, check if we can start another round
            # with unmapped low-conf frames as new single-frame clusters
            if not unmapped_low_conf_frames:
                print(f"Round {round_num} complete. No unmapped frames to retry.")
                break

            # Check if we made progress compared to the previous round
            current_unmapped_count = len(unmapped_low_conf_frames)
            if prev_unmapped_count is not None and current_unmapped_count >= prev_unmapped_count:
                print(f"Round {round_num} complete. No progress ({current_unmapped_count} unmapped frames, same as previous round). Moving to final re-mapping.")
                break
            prev_unmapped_count = current_unmapped_count

            # Create new single-frame clusters from unmapped low-conf frames
            new_clusters_added = False
            for frame_idx in list(unmapped_low_conf_frames):
                # Only add if not already mapped as a keyframe
                if frame_idx not in self.keyframe_memory.kf_idx:
                    clusters[frame_idx] = []  # Single-frame cluster
                    rest_clusters_kf_indices.append(frame_idx)
                    new_clusters_added = True

            if not new_clusters_added or len(rest_clusters_kf_indices) == 0:
                print(f"Round {round_num} complete. No new clusters can be added from unmapped frames.")
                break

            print(f"Round {round_num} complete. Retrying {len(rest_clusters_kf_indices)} unmapped frames as new clusters.")
            unmapped_low_conf_frames.clear()


        # Map any remaining unmapped low-confidence frames
        self.remap_unmapped_frames(images, distance_matrix, unmapped_low_conf_frames)
        print("Coarse registration complete.")


    def keyframe_mapping(self, images):
        '''
        Refine keyframe predictions via BFS traversal over KF pose graph.
        '''

        global_kf_idx = self.keyframe_memory.kf_idx
        global_kf_poses = self.keyframe_memory.poses[global_kf_idx]
        global_kf_dists = extrinsic_distance_batch_query(global_kf_poses, global_kf_poses)
        mean_kf_conf = self.keyframe_memory.conf.mean(dim=(1,2))

        num_kfs = len(global_kf_idx)
        visited_kf_local = set() # Stores local indices: 0 to num_kfs-1
        pq = [] # (priority, kf_local_idx)
        pq_set = set()

        start_kf_local_idx = 0
        global_start_idx = global_kf_idx[start_kf_local_idx]
        start_conf = mean_kf_conf[global_start_idx].item()
        heapq.heappush(pq, (-start_conf, start_kf_local_idx))
        pq_set.add(start_kf_local_idx)

        global_to_local_kf_idx = {global_idx: local_idx for local_idx, global_idx in enumerate(global_kf_idx)}

        pbar = tqdm(total=num_kfs, desc="Refining KeyFrames (BFS)")
        while len(visited_kf_local) < num_kfs:
            if not pq:
                # --- Handle disconnected components ---
                print("  > Priority queue empty, searching for new component...")
                best_unvisited_conf = -float('inf')
                start_kf_local_idx_new = -1 
                
                for kf_local_i in range(num_kfs):
                    if kf_local_i not in visited_kf_local:
                        global_idx = global_kf_idx[kf_local_i]
                        conf = mean_kf_conf[global_idx].item() 
                        if conf > best_unvisited_conf:
                            best_unvisited_conf = conf
                            start_kf_local_idx_new = kf_local_i
                
                if start_kf_local_idx_new == -1:
                    print("  > No unvisited keyframes left.")
                    break 

                global_start_idx_new = global_kf_idx[start_kf_local_idx_new]
                heapq.heappush(pq, (-best_unvisited_conf, start_kf_local_idx_new))
                pq_set.add(start_kf_local_idx_new)
                print(f"  > Seeding queue with new KF component starting at {global_start_idx_new} (conf: {best_unvisited_conf:.4f})")
                
            
            while pq:
                priority, current_kf_local_idx = heapq.heappop(pq)

                if current_kf_local_idx in pq_set:
                    pq_set.remove(current_kf_local_idx)

                current_global_idx = global_kf_idx[current_kf_local_idx]
                dists_to_current = global_kf_dists[current_kf_local_idx] 
                mask = (dists_to_current <= self.cfg.max_kf_search_distance) & (dists_to_current > 1e-6)
                anchor_local_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)

                if len(anchor_local_indices) == 0:
                    anchor_local_indices = torch.topk(dists_to_current, k=1, largest=False).indices
                
                else:
                    # sorted by distance
                    dists_for_indices = dists_to_current[anchor_local_indices]
                    anchor_local_indices = anchor_local_indices[torch.argsort(dists_for_indices)]
                    if len(anchor_local_indices) > self.cfg.max_kf_per_refinement:
                        anchor_local_indices = anchor_local_indices[:self.cfg.max_kf_per_refinement]
                
                anchor_global_indices = [global_kf_idx[i] for i in anchor_local_indices.tolist()]

                idx_to_map = [current_global_idx] + anchor_global_indices
                views_to_map = {
                        'images': images[:, idx_to_map].to(self.model.device),
                        'kf_idx': [current_global_idx],
                    }

                res = self.local_mapping(views_to_map, self.cfg, keyframe_memory=self.keyframe_memory)

                prediction_local = self.build_prediction_dict(res, idx_to_map)
                # TODO: rearrange based on conf and then align
                self.keyframe_memory.update(prediction_local, align=True, num_kf=len(idx_to_map), add_kf=False)
                

                visited_kf_local.add(current_kf_local_idx)
                pbar.update(1)

                for global_anchor_idx in anchor_global_indices:
                    # Convert global index back to local kf index
                    local_anchor_idx = global_to_local_kf_idx[global_anchor_idx]
                    
                    # Add to queue if not visited and not already in queue
                    if local_anchor_idx not in visited_kf_local and local_anchor_idx not in pq_set:
                        anchor_conf = mean_kf_conf[global_anchor_idx].item()
                        heapq.heappush(pq, (-anchor_conf, local_anchor_idx))
                        pq_set.add(local_anchor_idx)


            print(f"Refined {len(visited_kf_local)}/{num_kfs} keyframes.")
        
        pbar.close()
        print("Keyframe refinement complete.")


    def non_keyframe_mapping(self, images):
        '''
        Refine non-keyframe predictions
        '''
        
        Bs, num_frames, C, H, W = images.shape
        global_kf_idx = self.keyframe_memory.kf_idx
        refined_kf_poses = self.keyframe_memory.poses[global_kf_idx]
        non_kf_indices = [i for i in range(num_frames) if i not in global_kf_idx and i not in self.keyframe_memory.unmapped_frames]
        non_kf_local_to_global = non_kf_indices # List mapping
        non_kf_global_to_local = {g_idx: i for i, g_idx in enumerate(non_kf_indices)}
        all_non_kf_poses = self.keyframe_memory.poses[non_kf_indices]
        
        conf_mean_non_kf = self.keyframe_memory.conf[non_kf_indices].mean(dim=(1,2))
        sorted_non_kf = sorted(zip(non_kf_indices, conf_mean_non_kf.tolist()), key=lambda x: x[1], reverse=True)

        visited_non_kf = set()
        pbar_non_kf = tqdm(sorted_non_kf, desc="Refining Non-KeyFrames")

        for non_kf_idx, _ in pbar_non_kf:
            # 1. Get the current (coarse) pose of the non-KF
            if non_kf_idx in visited_non_kf:
                continue
            

            # 2. Find *Non-KeyFrame* neighbors
            current_non_kf_pose = self.keyframe_memory.poses[non_kf_idx].unsqueeze(0)
            dists_to_all_non_kf = extrinsic_distance_batch_query(current_non_kf_pose, all_non_kf_poses).squeeze(0)

            mask_dist = (dists_to_all_non_kf <= self.cfg.non_kf_search_window) & (dists_to_all_non_kf > 1e-6)
            neighbor_local_indices_in_window = torch.nonzero(mask_dist, as_tuple=False).squeeze(1)
            neighbor_global_indices = [non_kf_local_to_global[i.item()] for i in neighbor_local_indices_in_window]

            if neighbor_global_indices:
                neighbor_dists = [dists_to_all_non_kf[non_kf_global_to_local[g_idx]].item() for g_idx in neighbor_global_indices]
                sorted_neighbors = sorted(zip(neighbor_global_indices, neighbor_dists), key=lambda x: x[1])
                k_to_find_non_kf = self.cfg.max_kf_per_refinement // 2
                neighbor_global_indices = [g_idx for g_idx, dist in sorted_neighbors[:k_to_find_non_kf]]
            
            # 3. Form the non-KF window
            non_kf_window_indices = [non_kf_idx] + neighbor_global_indices
            non_kf_window_poses = self.keyframe_memory.poses[non_kf_window_indices]

            # 4. Find KeyFrame anchors based on *all* poses in the non-KF window
            dists_to_kfs = extrinsic_distance_batch_query(non_kf_window_poses, refined_kf_poses) # Shape: (window_size, num_kfs)
            
            # Find the minimum distance from *any* frame in the window to *each* KF
            if dists_to_kfs.numel() > 0:
                min_dists_to_kfs = dists_to_kfs.min(dim=0).values # Shape: (num_kfs)
            else:
                min_dists_to_kfs = torch.tensor([float('inf')])


            mask_kf = (min_dists_to_kfs <= self.cfg.max_kf_search_distance)
            anchor_local_indices = torch.nonzero(mask_kf, as_tuple=False).squeeze(1)


            if anchor_local_indices.numel() == 0:
                print("  > No KF anchors found within distance, selecting closest KF.")
                anchor_local_indices = torch.topk(min_dists_to_kfs, k=1, largest=False).indices
            
            else:
                # Sort by distance and truncate
                dists_for_indices = min_dists_to_kfs[anchor_local_indices]
                anchor_local_indices = anchor_local_indices[torch.argsort(dists_for_indices)]
                if len(anchor_local_indices) > self.cfg.max_kf_per_refinement - len(non_kf_window_indices):
                    anchor_local_indices = anchor_local_indices[:self.cfg.max_kf_per_refinement - len(non_kf_window_indices)]
                
                anchor_global_indices = [global_kf_idx[i] for i in anchor_local_indices.tolist()]

                if not anchor_global_indices:
                    raise RuntimeError("No anchors found for non-keyframe refinement after filtering.")


            idx_to_map = anchor_global_indices + non_kf_window_indices
            views_to_map = {
                    'images': images[:, idx_to_map].to(self.model.device),
                    'kf_idx': anchor_global_indices,
                }

            res = self.local_mapping(views_to_map, self.cfg, keyframe_memory=self.keyframe_memory)

            prediction_local = self.build_prediction_dict(res, idx_to_map)

            conf_local_nonkf = prediction_local['conf_sig'][len(anchor_global_indices):].mean(dim=(1,2))
            conf_global_nonkf = self.keyframe_memory.conf[non_kf_window_indices].mean(dim=(1,2))
            conf_mask = conf_local_nonkf > conf_global_nonkf

            # 5. Align local map using *only KF anchors* and fuse results
            self.keyframe_memory.update(prediction_local, align=True, num_kf=len(anchor_global_indices), add_kf=False)
            
            # 6. Mark all processed frames in the non-KF window as visited
            for g_idx in non_kf_window_indices:
                if conf_mask[non_kf_window_indices.index(g_idx)]:
                    visited_non_kf.add(g_idx)

        pbar_non_kf.close()
        print("Non-keyframe refinement complete.")


    def global_mapping(self, images):
        '''
        Run global refinement: refine keyframes first, then non-keyframes.
        '''

        for refine_iter in range(self.cfg.max_global_refinement_iters):
            print(f"Global refinement iteration {refine_iter+1}/{self.cfg.max_global_refinement_iters}")
            self.keyframe_mapping(images)
            self.non_keyframe_mapping(images)


    def run(self, images, poses_gt=None):
        '''
        Params:
            - images: (B, T, 3, H, W), in [-1, 1] range
            - poses_gt: (B, T, 4, 4), in world space
        '''

        # Let us SfM!
        assert images.min() >= -1 and images.max() <= 1, "Images should be in [-1, 1] range"

        Bs, T, _, H, W = images.shape

        self.keyframe_memory = SfMemory(self.cfg, T, H, W)
        if poses_gt is not None:
            self.keyframe_memory.set_gt_poses(poses_gt)

        feature_descriptors = self.extract_features(images)
        distance_matrix = get_distance_matrix(feature_descriptors, whitening=True)

        clusters = find_best_image_clustering(distance_matrix=distance_matrix,
                                    min_frames_per_cluster=self.cfg.min_cluster_size,
                                    max_frames_per_cluster=self.cfg.max_cluster_size
                                    )
        
        clusters, best_cluster_kf_idx = self.process_clusters(images, clusters, distance_matrix)

        # Stage 1: Initialization
        self.initialize_map(images, best_cluster_kf_idx, clusters[best_cluster_kf_idx])

        # Stage 2: Coarse registration
        rest_clusters_kf_indices = [kf for kf in clusters.keys() if kf != best_cluster_kf_idx]
        self.coarse_registration(images, distance_matrix, rest_clusters_kf_indices, clusters)

        # Stage 3: Global mapping
        self.global_mapping(images)

        return self.keyframe_memory

