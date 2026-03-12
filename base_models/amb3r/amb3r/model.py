# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/model.py

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .backend import BackEnd
from .frontend import FrontEnd
from .tools.pose_align import umeyama_alignment
from .tools.pose_interp import interpolate_poses


sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))
from moge.moge.train.losses import scale_invariant_alignment


class AMB3R(nn.Module):
    def __init__(self, device='cuda', metric_scale=True,
                 voxel_resolutions=[0.01], precision='bf16'):
        super().__init__()

        self.device = device
        self.name = 'amb3r'
        self.metric_scale = metric_scale
        self.voxel_resolutions = voxel_resolutions

        self.front_end = FrontEnd(metric_scale=metric_scale)
        self.backend = BackEnd(in_dim=2048+1024, 
                                  out_dim=1024, 
                                  k_neighbors=16)


    def prepare(self, data_type='bf16'):
        '''
        Prepare the model for data type conversion
        '''
        
        data_type_map = {
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32
        }

        if data_type not in data_type_map:
            raise ValueError(f"Unsupported data type: {data_type}")

        selected_dtype = data_type_map[data_type]
        self.front_end.model.aggregator.to(selected_dtype)
        for name, param in self.backend.named_parameters():
            if 'point_transformer' not in name:
                param.data = param.data.to(selected_dtype)


    def load_weights(self, path, data_type='bf16', strict=True):
        '''
        Model weights loading & data type preparation
        '''
        state_dict = torch.load(path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.load_state_dict(state_dict, strict=strict)
        self.prepare(data_type)


    def resize_feat(self, feat, scale_factor=None, target_size=None):
        '''
        Resize feature maps using bilinear interpolation.
        
        Params:
            - feat: input feature map of shape (B, T, H, W, C)
            - scale_factor: scaling factor for resizing (optional if target_size is provided)
            - target_size: target size for resizing (optional if scale_factor is provided)
        
        Returns:
            - resized_feat: resized feature map of shape (B, T, target_H, target_W, C)
        '''
        if scale_factor is None and target_size is None:
            raise ValueError("Either scale_factor or target_size must be provided")
        
        if scale_factor is not None:
            target_size = (int(feat.shape[2] * scale_factor), int(feat.shape[3] * scale_factor))

        Bs, T, H, W, C = feat.shape
        feat = F.interpolate(feat.permute(0, 1, 4, 2, 3).flatten(0, 1), size=target_size, mode='bilinear', align_corners=False)
        feat = feat.permute(0, 2, 3, 1).view(Bs, T, target_size[0], target_size[1], C)

        return feat


    def get_voxel_feat(self, res, detach=True):  
        '''
        Process front-end feature with backend to obtain voxel features
        
        Params:
            - res: dict containing front-end features and predictions
        
        Returns:
            - voxel_feat_aligned: raw voxel features in 2D space
            - voxel_feat_aligned_vis: voxel features in 2D space after zero conv and gating
            - voxel_layer_list: list of layers for processing voxel features

        '''
        feat = torch.cat([res['enc'], res['dec']], dim=-1)
        pts = res['world_points']  # (B, nimgs, H, W, 3)

        bs, nimgs, H, W, _ = pts.shape

        if detach:
            feat = feat.detach()
            pts = pts.detach()
        Bs, T, H, W, C = pts.shape
        feat = feat.view(Bs, T, H // 14, W // 14, feat.shape[-1])  # Reshape to (B, nimgs, H//14, W//14, 2048)
        feat = self.backend.aligner(feat)

        pts = self.resize_feat(pts, target_size=(H//7, W//7))  # Resize to 1/4 of original size
        Bs, T, Hs, Ws, C = pts.shape

        feat = self.resize_feat(feat, target_size=(Hs, Ws))


        # Step 2: Back-end processing
        with torch.amp.autocast("cuda", enabled=True):
            voxel_feat = self.backend.forward(pts, feat, voxel_sizes=self.voxel_resolutions)

        
        voxel_feat_fine = voxel_feat[-1].reshape(Bs, T, Hs, Ws, -1)  # Reshape to (B, nimgs, N_voxels, C_out)
        voxel_feat_aligned = self.backend.downsample(voxel_feat_fine.permute(0, 1, 4, 2, 3).flatten(0, 1))  # Downsample to match front-end feature size

        # Step 3: Align voxel features with front-end features
        voxel_feat_aligned_vis = self.backend.zero_conv(voxel_feat_aligned) *self.backend.gate_scale#* self.backend.gate_scale

        voxel_layer_list = []
        for l_idx, layer in enumerate(self.backend.zero_conv_layers):
            voxel_layer_list.append(
                {
                    'layer': layer,
                    'H': H//14,
                    'W': W//14,
                    'gate_scale': self.backend.gate_scales[l_idx]
                }

            )

        voxel_feat_aligned_vis = voxel_feat_aligned_vis.view(Bs, T, -1, H//14, W//14).permute(0, 1, 3, 4, 2)
        
        
        return voxel_feat_aligned, voxel_feat_aligned_vis, voxel_layer_list
    

    def forward(self, frames, iters=1):
        '''
        Forward pass of AMB3R 1) front-end processing 2) back-end processing
        '''
        
        # Front-end processing
        res_all = []
        images, patch_tokens = self.front_end.encode_patch_tokens(frames)

        res = self.front_end.decode_patch_tokens_and_heads(images, patch_tokens)
        res_all.append(res)

        # Back-end processing
        for i in range(iters):
            voxel_feat_aligned, voxel_feat_aligned_vis, voxel_layer_list = self.get_voxel_feat(res_all[i])
            patch_tokens = self.front_end.add_voxel_feat_to_patch_tokens(patch_tokens, voxel_feat_aligned_vis)
            res_all.append(self.front_end.decode_patch_tokens_and_heads(images, patch_tokens, voxel_feat=voxel_feat_aligned, voxel_layer_list=voxel_layer_list))

        return res_all
    

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        '''
        RMVDB input adapter
        '''
        def select_by_index(l, idx):
            """Select an element from a list by an index. Supports data batches with different indices.

            Args:
                l (list): List with potentially batched data items.
                idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
                    Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
            """
            if isinstance(idx, int):
                ret = l[idx]
            else:
                indices = idx
                ret = []
                for batch_idx, idx in enumerate(indices):
                    ret.append(l[idx][batch_idx])

                if isinstance(ret[0], np.ndarray):
                    ret = np.stack(ret, 0)
                else:
                    ret = torch.stack(ret, 0)

            return ret

        def exclude_index(l, exclude_idx):
            """Selects all element from a list, excluding a specific index. Supports data batches with different indices.

            Args:
                l (list): List with potentially batched data items.
                idx: idx can be an integer in case of non-batched data or in case samples in the batch have the same index.
                    Alternatively, idx can be an iterable that contains indices for each sample in the batch separately.
            """
            if isinstance(exclude_idx, int):
                ret = [ele for idx, ele in enumerate(l) if idx != exclude_idx]
            else:
                exclude_indices = exclude_idx
                ret = []
                for batch_idx, exclude_idx in enumerate(exclude_indices):
                    ret.append([ele[batch_idx] for idx, ele in enumerate(l) if idx != exclude_idx])

                transposed = list(zip(*ret))
                if isinstance(transposed[0][0], np.ndarray):
                    ret = [np.stack(ele, 0) for ele in transposed]
                else:
                    ret = [torch.stack(ele, 0) for ele in transposed]

            return ret

        
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        images = [image_key] + images_source
        images = np.stack(images, 1) / 255.0  # Normalize images to [0, 1]

        images = torch.from_numpy(images).float()
        images = (images - 0.5) * 2.0  # Normalize to [-1, 1]
        images = images.unsqueeze(0).permute(0, 2, 4, 1, 3) # 1, n, 3, H, W
        b, s, c, h, w = images.shape
        images = images.view(b * s, c, h, w)
        images_interpolated = F.interpolate(images, 
                                            size=(392, 518), 
                                            mode='bilinear', 
                                            align_corners=False)
        target_h, target_w = images_interpolated.shape[2:]
        images = images_interpolated.view(b, s, c, target_h, target_w)

        frames_ = {
            'images': images.to(self.device),
            'keyview_idx': keyview_idx
        }

        frames = {
            'frames': frames_,
        }

        return frames
    

    def output_adapter(self, model_output):
        '''
        RMVDB output adapter
        '''
        if self.metric_scale:
            depth = model_output[-1]['depth_metric']
        else:
            depth = model_output[-1]['depth']


        depth_key = depth[:, 0, ...].permute(0, 3, 1, 2)


        aux = {}

        pred = {
            'depth': depth_key.cpu().numpy(),
        }

        return pred, aux


    def run_backend(self, res, images, patch_tokens):
        '''
        run the back-end of AMB3R given front-end results and input images
        '''
        voxel_feat_aligned, voxel_feat_aligned_vis, voxel_layer_list = self.get_voxel_feat(res)
        patch_tokens = self.front_end.add_voxel_feat_to_patch_tokens(patch_tokens, voxel_feat_aligned_vis)

        res1 = self.front_end.decode_patch_tokens_and_heads(images, patch_tokens, voxel_feat=voxel_feat_aligned, voxel_layer_list=voxel_layer_list)

        return res1
    

    @torch.inference_mode()
    def run_amb3r_vo(self, frames, cfg, keyframe_memory):
        '''
        AMB3R-VO interface
        '''
        res_all = []
        
        images, patch_tokens = self.front_end.encode_patch_tokens(frames)
        res = self.front_end.decode_patch_tokens_and_heads(images, patch_tokens)
        res_all.append(res)


        conf = res['world_points_conf']  # (B, nimgs, H, W, 1)
        if conf.mean() > cfg.conf_threshold_front:
            return res_all[-1]
        
        
        

        res1 = self.run_backend(res, images, patch_tokens=patch_tokens)
        res_all.append(res1)

        compare=False
        blend = cfg.blend

        if blend:
            points0 = res_all[0]['world_points']  # (B, nimgs, H, W, 3)
            conf0 = res_all[0]['world_points_conf']  # (B, n
            c2w0 = res_all[0]['pose'][0]

            points1 = res_all[-1]['world_points']  # (B, nimgs, H, W, 3)
            conf1 = res_all[-1]['world_points_conf']  # (B, nimgs, H, W, 1)
            c2w1 = res_all[-1]['pose'][0]


            max_median_ratio_0 =  (points0.norm(dim=-1).max() / points0.norm(dim=-1).median()).item()
            max_median_ratio_1 =  (points1.norm(dim=-1).max() / points1.norm(dim=-1).median()).item()

            if conf1.mean() < conf0.mean() + 0.2 and conf1.mean() < 1.5:
                compare=True

            Bs, T, H, W, _ = points0.shape

            if conf1.mean() >= 1 + np.exp(1.0):
                blend=False


            # OK, AMB3R is not confident enough as well, marry them
            conf_sum = conf0 + conf1

            points0, scale0 = scale_invariant_alignment(
                                            points0.contiguous().view(1, T* H, W, 3),
                                            points1.contiguous().view(1, T* H, W, 3),
                                            (conf_sum >= torch.quantile(conf_sum, 0.5)).view(1, T* H, W),
                                            trunc=1.0)
            points0 = points0.view(Bs, T, H, W, 3)


            cross_consistency = torch.abs((points0 - points1)).mean()

            if cross_consistency > 0.04 and max_median_ratio_1 / max_median_ratio_0 > 1.25 and max_median_ratio_1 > 5.0:
                # print("Backend is too radical, let us use front-end here")
                return res_all[0]
        

            if blend and cross_consistency < 0.04:
                points_final = (points0 * conf0[..., None] + points1 * conf1[..., None]) / (conf_sum[..., None] + 1e-8)
                c2w0[:, :3, 3] *= scale0

                c2w_blend = interpolate_poses(c2w0.view(-1, 4, 4), c2w1.view(-1, 4, 4), 
                                            conf0.view(-1, H, W), conf1.view(-1, H, W))
                c2w_blend = c2w_blend.view(Bs, T, 4, 4).to(points0.device)

                res_new = {
                    'world_points': points_final,
                    'world_points_conf': conf_sum / 2.0,
                    'pose': c2w_blend,
                }
                res_all.append(res_new)
        
        if keyframe_memory is None:
            return res_all[-1]
        
        ref_pts = keyframe_memory.pts[keyframe_memory.cur_kf_idx]
        ref_conf = keyframe_memory.conf[keyframe_memory.cur_kf_idx]
        
        if cross_consistency > 0.04 or compare:
            best_rel_err = 1e5
            best_res = None
            for i, res in enumerate(res_all):
                pts_pred = res['world_points'][0, :len(ref_pts)]
                conf = res['world_points_conf'][0, :len(ref_pts)]
                conf_sig = (conf-1)/ conf
                if ref_conf is not None:
                    mask = ref_conf > torch.quantile(ref_conf, 0.5)
                    # mask = ref_conf * conf_sig.cpu()
                    # mask = mask > torch.quantile(mask, 0.5)
                    pts_pred = pts_pred[mask]
                    ref_pts_masked = ref_pts[mask]
                else:
                    ref_pts_masked = ref_pts
                aligned_pts, info = umeyama_alignment(pts_pred.reshape(-1, 3).cpu(), ref_pts_masked.reshape(-1, 3).cpu())

                rel_err = torch.norm(aligned_pts - ref_pts_masked.reshape(-1, 3).cpu(), dim=-1) / torch.norm(ref_pts_masked.reshape(-1, 3).cpu(), dim=-1)
                # rel_err *= ref_conf[mask].cpu()
                rel_err = rel_err.mean().item()
                if rel_err < best_rel_err:
                    best_rel_err = rel_err
                    best_res = res
                    best_idx = i            
            
            res_all.append(best_res)


        return res_all[-1]
    

    @torch.inference_mode()
    def extract_amb3r_sfm_features(self, views):
        '''
        AMB3R-SfM interface - feature extraction for image clustering
        '''
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, patch_tokens = self.front_end.encode_patch_tokens(views)
            descriptors_chunk = patch_tokens['x_norm_patchtokens'].mean(dim=1) # Shape: (chunk, C)
        
        return descriptors_chunk
    

    @torch.inference_mode()
    def run_amb3r_sfm(self, frames, cfg, keyframe_memory=None, benchmark_conf0=None):
        '''
        AMB3R-SfM interface, benchmark_conf0 is for AMB3R base model for early stopping
        '''
        res_all = []
        # Step 1: Front-end processing
        
        # with torch.no_grad():
        images, patch_tokens = self.front_end.encode_patch_tokens(frames)

        res = self.front_end.decode_patch_tokens_and_heads(images, patch_tokens)
        conf_0 = res['world_points_conf'][0].mean(dim=(1, 2)).cpu()
        res['benchmark_conf0'] = conf_0
        res_all.append(res)

        if benchmark_conf0 is not None:
            conf_mean = res['world_points_conf'][0].mean(dim=(1, 2)).cpu() # (T, )
            # Check if any frame has confidence higher than benchmark_conf0
            if not (conf_mean > benchmark_conf0 - 1e-2).any():
                return None

        # Step 2: Back-end processing
        voxel_feat_aligned, voxel_feat_aligned_vis, voxel_layer_list = self.get_voxel_feat(res_all[0], detach=True)
        patch_tokens = self.front_end.add_voxel_feat_to_patch_tokens(patch_tokens, voxel_feat_aligned_vis)

        # Step 3: Decode patch tokens and heads
        res1 = self.front_end.decode_patch_tokens_and_heads(images, patch_tokens, voxel_feat=voxel_feat_aligned, voxel_layer_list=voxel_layer_list)

        res1['benchmark_conf0'] = conf_0

        if res1['world_points_conf'].mean() < res_all[0]['world_points_conf'].mean():   
            # For really really bad cases, just use the first result
            print(f'Unusual bad case detected in backend, skip it. conf: {res1["world_points_conf"].mean().item():.4f} vs {res_all[0]["world_points_conf"].mean().item():.4f}')
            return res_all[-1]
        
        res_all.append(res1)

        return res_all[-1]
    
    
    def run_amb3r_benchmark(self, frames):
        '''
        Forward pass of AMB3R 1) front-end processing 2) back-end processing
        '''
        
        # Front-end processing
        images, patch_tokens = self.front_end.encode_patch_tokens(frames)

        res_f = self.front_end.decode_patch_tokens_and_heads(images, patch_tokens)

        # Back-end processing
        voxel_feat_aligned, voxel_feat_aligned_vis, voxel_layer_list = self.get_voxel_feat(res_f)
        patch_tokens = self.front_end.add_voxel_feat_to_patch_tokens(patch_tokens, voxel_feat_aligned_vis)
        res_b = self.front_end.decode_patch_tokens_and_heads(images, patch_tokens, voxel_feat=voxel_feat_aligned, voxel_layer_list=voxel_layer_list)

        out = {
            'world_points': res_b['world_points'],
            'depth': res_b['depth'],
            'pose': res_b['pose'],
            'pts3d_by_unprojection': res_b['pts3d_by_unprojection'],
        }

        return out




