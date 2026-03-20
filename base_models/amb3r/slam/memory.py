# References:
#   https://github.com/HengyiWang/amb3r/blob/main/slam/memory.py

import torch

from amb3r.tools.pose_interp import interpolate_poses
from amb3r.tools.pts_align import  coordinate_alignment
from amb3r.tools.keyframes import select_keyframes_iteratively
from amb3r.tools.pose_dist import extrinsic_distance_batch_query


class SLAMemory():
    def __init__(self, cfg, num_frames, H, W):
        self.cfg = cfg
        self.num_frames = num_frames
        self.pts = torch.zeros((num_frames, H, W, 3))
        self.conf = torch.zeros((num_frames, H, W))
        self.poses = torch.zeros((num_frames, 4, 4))
        self.iter = torch.zeros((num_frames))
        self.kf_idx = None
        self.cur_kf_idx = None

    
    def initialize(self, res):
        '''
        Initializes the memory/map with first batch of predictions
        '''
        
        conf = res['world_points_conf'][0].cpu() # (T, H, W)
        pts = res['world_points'][0].cpu() # (T, N, 3)
        c2w = res['pose'][0].cpu() # (T, 4, 4)
        conf_sig = (conf - 1) / conf
        conf_sig[conf_sig == 0] = 1e-6
        dists = extrinsic_distance_batch_query(c2w, c2w)

        is_keyframe = select_keyframes_iteratively(dists, conf, self.cfg.keyframe_threshold,
                                                   keyframe_indices=[0])
        keyframe_indices = torch.nonzero(is_keyframe, as_tuple=False).squeeze(1)

        self.pts[:len(pts)] = pts
        self.conf[:len(pts)] = conf_sig
        self.poses[:len(pts)] = c2w
        self.iter[:len(pts)] = 1
        self.kf_idx = keyframe_indices
        self.cur_kf_idx = keyframe_indices

    
    def resample_keyframes(self, current_kf_indices, current_poses, num_to_keep,
                            all_mapped_poses, num_top_k=3, sum_min=True):
        """
        Keyframe management strategy
        
        Params:
            - current_kf_indices: tensor of global indices of active keyframes
            - current_poses: tensor of poses corresponding to active keyframes
            - num_to_keep: number of keyframes to keep after resampling
            - all_mapped_poses: tensor of all poses in the map for distance calculations
            - num_top_k: number of adjacent candidates to keep  
        """
        if len(current_kf_indices) <= num_to_keep:
            return current_kf_indices

        dists = extrinsic_distance_batch_query(current_poses, current_poses)
        position_map = {global_idx.item(): pos for pos, global_idx in enumerate(current_kf_indices)}

        newest_kf_global_idx = current_kf_indices[-1].item()
        resampled_indices = [newest_kf_global_idx]
        
        initial_candidates = current_kf_indices[:-1]
        if len(initial_candidates) > 0:
            newest_kf_pos = position_map[newest_kf_global_idx]
            initial_candidate_positions = [position_map[idx.item()] for idx in initial_candidates]
            dists_to_newest = dists[initial_candidate_positions, newest_kf_pos]
            sorted_dist_indices = torch.argsort(dists_to_newest)
            sorted_candidates = initial_candidates[sorted_dist_indices]

            for candidate_kf in sorted_candidates:
                if len(resampled_indices) >= num_top_k + 1: break
                candidate_global_idx = candidate_kf.item()
                is_loop_closure = abs(candidate_global_idx - newest_kf_global_idx) > 200
                if is_loop_closure:
                    resampled_indices.append(candidate_global_idx)
                    continue
                is_diverse = True
                candidate_pos = position_map[candidate_global_idx]
                for selected_idx in resampled_indices:
                    if dists[candidate_pos, position_map[selected_idx]] <= self.cfg.keyframe_resample_threshold_min:
                        is_diverse = False
                        break
                if is_diverse:
                    resampled_indices.append(candidate_global_idx)

        candidate_pool = sorted([idx.item() for idx in current_kf_indices if idx.item() not in resampled_indices])

        # Ensure we leave space for the bridge frames
        num_to_fill = num_to_keep - self.cfg.bridge_keyframes
        while len(resampled_indices) < num_to_fill:
            best_candidate_to_add = None
            for candidate_global_idx in candidate_pool:
                candidate_pos = position_map[candidate_global_idx]
                min_dist = min(dists[candidate_pos, position_map[s_idx]] for s_idx in resampled_indices)
                max_dist = max(dists[candidate_pos, position_map[s_idx]] for s_idx in resampled_indices)
                if min_dist <= self.cfg.keyframe_resample_threshold and max_dist <= self.cfg.keyframe_resample_threshold_max_b:
                    best_candidate_to_add = candidate_global_idx
                    break
            if best_candidate_to_add is None: break
            resampled_indices.append(best_candidate_to_add)
            candidate_pool.remove(best_candidate_to_add)


        if self.cfg.bridge_keyframes > 0:            
            resampled_set = set(resampled_indices)
            middle_candidate_pool = [
                idx.item() for idx in current_kf_indices if idx.item() not in resampled_set
            ]

            if len(middle_candidate_pool) > 0:
                middle_candidate_poses = all_mapped_poses[middle_candidate_pool]
                resampled_poses = all_mapped_poses[resampled_indices]
                dists_middle_to_resampled = extrinsic_distance_batch_query(
                    middle_candidate_poses, resampled_poses
                )

                if not sum_min:
                    min_dists, _ = torch.min(dists_middle_to_resampled, dim=1)
                    k = min(self.cfg.bridge_keyframes, len(middle_candidate_pool))
                    _, top_k_indices_in_pool = torch.topk(min_dists, k=k, largest=False)

                else:
                    sum_dists_per_candidate = torch.sum(dists_middle_to_resampled, dim=1)
                    k = min(self.cfg.bridge_keyframes, len(middle_candidate_pool))
                    _, top_k_indices_in_pool = torch.topk(sum_dists_per_candidate, k=k, largest=False)
                
                bridge_frames_to_add = torch.tensor(middle_candidate_pool)[top_k_indices_in_pool]

        final_indices = torch.tensor(bridge_frames_to_add.tolist() + sorted(list(set(resampled_indices))), dtype=torch.long)
        return final_indices
    

    def keyframe_management(self):
        """
        Manages the active keyframes by resampling when necessary.
        """

        last_kf_idx = self.cur_kf_idx[-1]
        last_kf_pose = self.poses[last_kf_idx][None]

        other_kf_idx = self.cur_kf_idx[self.cur_kf_idx != last_kf_idx]
        other_kf_pose = self.poses[other_kf_idx]

        dists_to_others = extrinsic_distance_batch_query(last_kf_pose, other_kf_pose)
        max_dist = torch.max(dists_to_others)

        if max_dist < self.cfg.keyframe_resample_threshold_max_f and len(self.cur_kf_idx) <= self.cfg.max_keyframes:
            return
        

        print(f"Active keyframes ({len(self.cur_kf_idx)}) > max ({self.cfg.max_keyframes}). Resampling for loop closure...")

        current_active_kfs = self.kf_idx
        current_kf_poses = self.poses[current_active_kfs]
        all_poses = self.poses

        resampled_kfs = self.resample_keyframes(
                            current_active_kfs,
                            current_kf_poses,
                            num_to_keep=self.cfg.min_keyframes,
                            all_mapped_poses=all_poses,
                            num_top_k=self.cfg.top_k_keyframes
                        )
        
        print(f"Resampled to {len(resampled_kfs)} keyframes: {resampled_kfs.tolist()}")
        self.cur_kf_idx = resampled_kfs


    def update(self, res, start_idx, end_idx):
        '''
        Updates the memory/map with new batch of predictions and performs keyframe management.
        '''

        # TODO: Low confidence re-start removed in released version
        pts_local = res['world_points'][0].cpu() # (T, N, 3)
        conf_local = res['world_points_conf'][0].cpu() # (T, H, W)
        c2w_local = res['pose'][0].cpu() # (T, 4, 4)
        conf_sig_local = (conf_local - 1) / conf_local
        conf_sig_local[conf_sig_local == 0] = 1e-6

        map_idx = torch.cat([self.cur_kf_idx, torch.arange(start_idx, end_idx+1)], dim=0)
        pts_global = self.pts[map_idx]
        conf_global = self.conf[map_idx]
        c2w_global = self.poses[map_idx]
        iter_global = self.iter[map_idx]

        # Coordinate alignment
        pts_global_from_local, c2w_global_from_local = coordinate_alignment(pts_local, c2w_local, conf_sig_local, 
                                                                                 pts_global, c2w_global, conf_global, len(self.cur_kf_idx), 
                                                                                 transform=self.cur_kf_idx[0]!= 0,
                                                                                 scale=None) 

        # Map update
        conf_global_sum = conf_global * iter_global[:, None, None]

        self.pts[map_idx] = (conf_global_sum[..., None] * pts_global + 
                            conf_sig_local[..., None] * pts_global_from_local) / (conf_global_sum[..., None] + conf_sig_local[..., None])
        self.conf[map_idx] = (conf_global_sum + conf_sig_local) / (iter_global[:, None, None] + 1)
        self.iter[map_idx] = iter_global + 1

        c2w_global_merged = interpolate_poses(c2w_global, c2w_global_from_local, conf_global, conf_sig_local, interpolate=True)

        self.poses[map_idx] = c2w_global_merged


        # Keyframe discovery
        dists = extrinsic_distance_batch_query(c2w_global_merged, c2w_global_merged)
        is_keyframe = select_keyframes_iteratively(dists, conf_local, self.cfg.keyframe_threshold,
                                                   keyframe_indices=list(range(len(self.cur_kf_idx))))

        if is_keyframe[len(self.cur_kf_idx):].sum() > 0:
            new_keyframe_indices = torch.nonzero(is_keyframe[len(self.cur_kf_idx):], as_tuple=False).squeeze(1) + start_idx
            self.cur_kf_idx = torch.cat([self.cur_kf_idx, new_keyframe_indices], dim=0)
            self.kf_idx = torch.cat([self.kf_idx, new_keyframe_indices], dim=0)


            # Keyframe management
            self.keyframe_management()
            