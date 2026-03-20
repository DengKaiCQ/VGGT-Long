# References:
#   https://github.com/HengyiWang/amb3r/blob/main/sfm/memory.py

import torch
from amb3r.tools.pts_align import transform_pts_and_poses_global_to_local, robust_scale_invariant_alignment, coordinate_alignment
from amb3r.tools.pose_interp import interpolate_poses
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from amb3r.tools.pose_align import average_transforms_with_weights
from amb3r.tools.keyframes import select_keyframes_iteratively
from sfm.clustering import find_best_kf_clustering
from benchmark.tools.pose_eval import get_results_from_camera_pose


class SfMemory():
    def __init__(self, cfg, num_frames, H, W):
        self.cfg = cfg
        self.num_frames = num_frames
        self.pts = torch.zeros((num_frames, H, W, 3))
        self.conf = torch.zeros((num_frames, H, W))
        self.poses = torch.zeros((num_frames, 4, 4))
        self.iter = torch.zeros((num_frames))
        self.kf_idx = None
        self.kf_clusters = None
        self.unmapped_frames = set()


    def set_gt_poses(self, poses_gt):
        self.poses_gt = poses_gt
    
    def add_unmapped_frames(self, unmapped_frames):
        self.unmapped_frames.update(unmapped_frames)


    def initialize(self, cluster_pred_all, best_kf_idx):
        cluster_res_best = cluster_pred_all[best_kf_idx]

        T, H, W, _ = cluster_res_best['pred']['pts'].shape

        predictions = {
            'pts': cluster_res_best['pred']['pts'].cpu(),
            'pose': cluster_res_best['pred']['pose'].cpu(),
            'conf_sig': cluster_res_best['pred']['conf_sig'].cpu(),
            'idx': cluster_res_best['idx'],
            'iter': torch.ones(len(cluster_res_best['idx']))
        }

        remaining_chunks = sorted(
            cluster_pred_all.items(),
            key=lambda item: item[1]['pred']['conf'][0].mean(),
            reverse=True
        )

        for member_idx, member_data in remaining_chunks:
            member_pred = member_data['pred']
            member_idx_to_use = member_data['idx']

            member_idx_map = {global_id: local_idx
                              for local_idx, global_id in enumerate(member_idx_to_use)}
            mapping_indices = [member_idx_map[global_idx] for global_idx in cluster_res_best['idx']]

            pts_reordered = member_pred['pts'][mapping_indices]
            conf_sig_reordered = member_pred['conf_sig'][mapping_indices]
            pose_reordered = member_pred['pose'][mapping_indices]

            pts_reordered_canonical, pose_reordered_canonical = transform_pts_and_poses_global_to_local(pts_reordered, pose_reordered)

            pts_reordered_scaled, kf_scale = robust_scale_invariant_alignment(
                                             pts_reordered_canonical.contiguous().view(1, T * H, W, 3),
                                             predictions['pts'].contiguous().view(1, T * H, W, 3),
                                             (conf_sig_reordered > torch.quantile(conf_sig_reordered.contiguous(), 0.5)).view(1, T * H, W),
                                             trunc=None,
                                             robust=True)
            pts_reordered_scaled = pts_reordered_scaled.view(T, H, W, 3)
            pose_reordered_scaled = pose_reordered_canonical.clone()
            pose_reordered_scaled[:, :3, 3] *= kf_scale

            # merge with existing predictions
            conf_final = predictions['conf_sig'] * predictions['iter'][:, None, None]

            conf_sum = conf_final + conf_sig_reordered
            conf_sum[conf_sum == 0] = 1e-6

            predictions['pts'] = (predictions['pts'] * conf_final.unsqueeze(-1) + pts_reordered_scaled * conf_sig_reordered.unsqueeze(-1)) / conf_sum.unsqueeze(-1)

            pose_merged = interpolate_poses(
                predictions['pose'],
                pose_reordered_scaled,
                conf_final,
                conf_sig_reordered,
                interpolate=True
            )
            predictions['pose'] = pose_merged
            predictions['conf_sig'] = conf_sum / (predictions['iter'][:, None, None] + 1)
            predictions['iter'] += 1

        # Keyframe selection
        dists = extrinsic_distance_batch_query(predictions['pose'], predictions['pose'])

        is_keyframe = select_keyframes_iteratively(dists, predictions['conf_sig'],
                                                   self.cfg.keyframe_threshold, keyframe_indices=[0])
        keyframe_indices = torch.nonzero(is_keyframe, as_tuple=False).squeeze(1)

        kf_idx = [cluster_res_best['idx'][i] for i in keyframe_indices.tolist()]

        # Memory update
        self.pts[predictions['idx']] = predictions['pts']
        self.conf[predictions['idx']] = predictions['conf_sig']
        self.poses[predictions['idx']] = predictions['pose']
        self.iter[predictions['idx']] = predictions['iter']
        self.kf_idx = kf_idx
        self.kf_clusters = [kf_idx]


    def form_global_kf_clusters(self, kf_poses, kf_poses_idx, kf_conf):
        """
        Cluster keyframes by pose distance and return global-index clusters.
        """
        
        final_keyframes_dists = extrinsic_distance_batch_query(kf_poses, kf_poses)

        local_clusters_dict, unassigned_local_indices = find_best_kf_clustering(
            distance_matrix=final_keyframes_dists,
            conf=kf_conf,
            min_frames_per_cluster=self.cfg.min_keyframes_per_cluster,
            max_frames_per_cluster=self.cfg.max_keyframes_per_cluster,
            max_pose_distance=self.cfg.max_pose_distance_per_kf_cluster
        )

        global_kf_clusters = []

        for local_cluster_list in local_clusters_dict.values():
            global_cluster = [kf_poses_idx[local_idx] for local_idx in local_cluster_list]
            global_kf_clusters.append(global_cluster)

        for local_idx in unassigned_local_indices:
            global_kf_clusters.append([kf_poses_idx[local_idx]])

        return global_kf_clusters


    def update_global_kf_clusters(self, candidate_idx):
        """
        Update keyframe selection and clusters given new candidate frame indices.
        """
        
        search_kf_idx = self.kf_idx + candidate_idx
        pose_kf_candidates = self.poses[search_kf_idx]
        conf_kf_candidates = self.conf[search_kf_idx]

        dists = extrinsic_distance_batch_query(pose_kf_candidates, pose_kf_candidates)
        is_keyframe = select_keyframes_iteratively(dists, conf_kf_candidates, self.cfg.keyframe_threshold,
                                                   keyframe_indices=list(range(len(self.kf_idx))))

        keyframe_indices = torch.nonzero(is_keyframe, as_tuple=False).squeeze(1)
        new_kf_idx = [search_kf_idx[i] for i in keyframe_indices.tolist()[len(self.kf_idx):]]

        final_kf_idx = self.kf_idx + new_kf_idx

        print(f"Updated global keyframes: {final_kf_idx}")
        self.kf_idx = final_kf_idx

        final_kf_poses = self.poses[final_kf_idx]

        max_dist = 0.0
        if len(final_kf_idx) > 1:
            final_kf_dists = extrinsic_distance_batch_query(final_kf_poses, final_kf_poses)
            max_dist = final_kf_dists.max()

        if len(final_kf_idx) > self.cfg.max_mapping_frames or max_dist > self.cfg.max_pose_distance_per_kf_cluster:
            global_kf_clusters = self.form_global_kf_clusters(
                kf_poses=final_kf_poses,
                kf_poses_idx=final_kf_idx,
                kf_conf=self.conf[final_kf_idx]
            )
            print(global_kf_clusters)
        else:
            global_kf_clusters = [final_kf_idx]

        self.kf_clusters = global_kf_clusters


    def fuse_into_memory(self, idx, pts_aligned, c2w_aligned, conf_local):
        """
        Confidence-weighted fusion of local predictions into global memory.
        """
        
        pts_global = self.pts[idx]
        c2w_global = self.poses[idx]
        conf_global = self.conf[idx]
        iter_global = self.iter[idx]

        conf_global_sum = conf_global * iter_global[:, None, None]

        self.pts[idx] = (conf_global_sum[..., None] * pts_global +
                         conf_local[..., None] * pts_aligned) / (conf_global_sum[..., None] + conf_local[..., None])
        self.conf[idx] = (conf_global_sum + conf_local) / (iter_global[:, None, None] + 1)
        self.iter[idx] = iter_global + 1

        c2w_merged = interpolate_poses(c2w_global, c2w_aligned, conf_global, conf_local, interpolate=True)
        self.poses[idx] = c2w_merged


    def update_kf(self, res, map_idx, num_kf, adaptive=True, non_kf_conf_threshold=None):
        """
        Align local predictions to global frame, then fuse keyframes (and optionally non-KFs).
        Used during coarse registration
        """
        pts_local = res['world_points'][0].cpu()
        pose_local = res['pose'][0].cpu()
        conf_local = res['world_points_conf'][0].cpu()
        conf_sig_local = (conf_local - 1) / conf_local
        map_idx_ori = map_idx.copy()

        # Align local coordinate system to global
        pts_global_from_local, c2w_global_from_local = coordinate_alignment(
            pts_local, pose_local, conf_sig_local,
            self.pts[map_idx], self.poses[map_idx], self.conf[map_idx],
            num_kf=num_kf, transform=True, scale=None, trunc=1.0)

        # ── Fuse keyframes ──
        map_idx_kf = torch.tensor(map_idx[:num_kf])
        conf_sig_local_kf = conf_sig_local[:num_kf]
        pts_global_from_local_kf = pts_global_from_local[:num_kf]
        c2w_global_from_local_kf = c2w_global_from_local[:num_kf]

        if adaptive:
            # Only fuse frames where local confidence exceeds global
            conf_global_kf = self.conf[map_idx_kf]
            mask_kf = conf_sig_local_kf.mean(dim=(-1, -2)) > conf_global_kf.mean(dim=(-1, -2))

            if mask_kf.sum() > 0:
                self.fuse_into_memory(
                    map_idx_kf[mask_kf],
                    pts_global_from_local_kf[mask_kf],
                    c2w_global_from_local_kf[mask_kf],
                    conf_sig_local_kf[mask_kf])
        else:
            self.fuse_into_memory(
                map_idx_kf,
                pts_global_from_local_kf,
                c2w_global_from_local_kf,
                conf_sig_local_kf)

        # ── Merge high-confidence non-KF frames ──
        merged_non_kf_global = []
        if non_kf_conf_threshold is not None and len(map_idx) > num_kf:
            conf_sig_non_kf = conf_sig_local[num_kf:]
            mean_conf_non_kf = conf_sig_non_kf.mean(dim=(-1, -2))
            merge_mask = mean_conf_non_kf > non_kf_conf_threshold

            max_conf_kf = conf_sig_local_kf.mean(dim=(-1, -2)).max()

            if merge_mask.sum() > 0 and max_conf_kf > non_kf_conf_threshold:
                non_kf_map_idx = torch.tensor(map_idx[num_kf:])
                merge_idx = non_kf_map_idx[merge_mask]

                self.fuse_into_memory(
                    merge_idx,
                    pts_global_from_local[num_kf:][merge_mask],
                    c2w_global_from_local[num_kf:][merge_mask],
                    conf_sig_non_kf[merge_mask])

                merged_non_kf_global = merge_idx.tolist()
                print(f"  [update_kf] Merged {len(merged_non_kf_global)} high-conf non-KF frames: {merged_non_kf_global}")

                # Allow merged non-KF frames to become keyframes
                self.update_global_kf_clusters(merged_non_kf_global)

        aligned_res = {
            'pts': pts_global_from_local,
            'pose': c2w_global_from_local,
            'conf_sig': conf_sig_local,
            'idx': map_idx_ori,
            'iter': torch.ones(len(map_idx_ori)),
        }

        return aligned_res, merged_non_kf_global


    def update(self, res, align=False, num_kf=None, add_kf=True, adaptive=False):
        """
        Fuse a prediction dict into global memory, optionally with alignment.
        """
        
        pts_local = res['pts']
        c2w_local = res['pose']
        conf_sig_local = res['conf_sig'] + 1e-6
        map_idx = res['idx']

        # Optional coordinate alignment
        if not align:
            pts_aligned = pts_local
            c2w_aligned = c2w_local
        else:
            pts_aligned, c2w_aligned = coordinate_alignment(
                pts_local, c2w_local, conf_sig_local,
                self.pts[map_idx], self.poses[map_idx], self.conf[map_idx],
                num_kf=num_kf, transform=True, scale=None, trunc=1.0)

        # Determine which frames to update
        if adaptive:
            map_idx_t = torch.tensor(map_idx)
            conf_global_existing = self.conf[map_idx_t]
            mask = conf_sig_local.mean(dim=(-1, -2)) >= conf_global_existing.mean(dim=(-1, -2))

            if mask.sum() == 0:
                if not add_kf:
                    return
                # Skip fusion but still do keyframe selection below
                map_idx_update = map_idx_t[mask]  # empty
            else:
                map_idx_update = map_idx_t[mask]
                pts_aligned = pts_aligned[mask]
                c2w_aligned = c2w_aligned[mask]
                conf_sig_local = conf_sig_local[mask]
        else:
            map_idx_update = torch.tensor(map_idx)

        # Fuse into global memory
        if len(map_idx_update) > 0:
            self.fuse_into_memory(map_idx_update, pts_aligned, c2w_aligned, conf_sig_local)

        if hasattr(self, 'poses_gt'):
            poses_pred = self.poses
            poses_gt = self.poses_gt
            valid_mask = poses_pred.abs().sum(dim=(1, 2)) > 1e-6
            result, extri, gt_extri = get_results_from_camera_pose(poses_pred[valid_mask], poses_gt[valid_mask])
            print(f"Updated memory with results: {result}")

        if not add_kf:
            return

        # Update keyframe clusters
        self.update_global_kf_clusters(map_idx)
