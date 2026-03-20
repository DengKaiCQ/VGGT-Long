
# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/backend.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean
from pytorch3d.ops import knn_points, knn_gather

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

from .blocks import ZeroConvBlock, DownBlock
from .tools.voxel_utils import get_vox_indices
from ptv3.point_transformer import PointTransformerV3




class BackEnd(nn.Module):
    def __init__(self, hash_base=1024, in_dim=1024, out_dim=256, 
                 k_neighbors=16, depth=48):
        super(BackEnd, self).__init__()
        self.base = hash_base
        self.aligner = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.GELU(),
            nn.Linear(in_dim//2, out_dim),
            nn.GELU()
        )

        self.point_transformer = PointTransformerV3()
        self.k_neighbors = k_neighbors
        self.downsample = DownBlock(in_channels=1024, mid_channels=1024, out_channels=1024)
        self.zero_conv = ZeroConvBlock()
        self.gate_scale = nn.Parameter(torch.ones(1))

        self.zero_conv_layers = nn.ModuleList(
            [ZeroConvBlock() for _ in range(depth)]
        )
        self.gate_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(depth)]
        )


    @torch.no_grad()
    def hash_fn(self, coords):
        '''
        A simple hash function for voxel coordinates
        '''
        b, x, y, z = coords.unbind(dim=1)
        return ((b.long() << 48)
               | (x.long() << 32)
               | (y.long() << 16)
               |  z.long())
    
    
    def mean_by_voxel(self, points, feats, batch_ids, voxel_size, bounding_boxes):
        '''Compute mean features for each voxel.
        
        Params:
            - points: (N, 3) tensor of point coordinates
            - feats: (N, C) tensor of point features
            - batch_ids: (N,) tensor of batch indices for each point
            - voxel_size: scalar or (3,) tensor defining the size of each voxel
            - bounding_boxes: (B, 2, 3) tensor of min and max coordinates for each batch
        
        Returns:
            - voxel_feats: (M, C) tensor of mean features for each voxel
            - info: dict containing 'unique_indices' which are the voxel indices corresponding to the mean features
        
        '''
        
        voxel_indices = get_vox_indices(points, batch_ids, voxel_size, bounding_boxes, shift=False, cat_batch_ids=True)
        voxel_hash = self.hash_fn(voxel_indices)
        unique_hash, inverse_id = torch.unique(voxel_hash, return_inverse=True)

        voxel_feats = scatter_mean(feats, inverse_id, dim=0) 
        original_indices = torch.arange(voxel_hash.shape[0], device=voxel_hash.device)
        min_original_indices_per_unique_id = torch.full((unique_hash.shape[0],),
                                                voxel_hash.shape[0],
                                                dtype=torch.long,
                                                device=voxel_hash.device)
        
        first_occurrence_original_indices = torch.scatter_reduce(
            min_original_indices_per_unique_id,
            0,
            inverse_id,
            original_indices,
            reduce="amin",
            include_self=False
        )

        unique_voxel_indices = voxel_indices[first_occurrence_original_indices]

        info = {
            'unique_indices': unique_voxel_indices,
        }    

        return voxel_feats, info

    
    def voxel_to_point_interpolation(self, point_out, pts, chunk_size=50000):
        """Interpolate point/voxel features back to exact continuous points via batched chunked KNN."""
        Bs = pts.shape[0] if len(pts.shape) == 3 else (pts.shape[0] // pts.shape[1] if hasattr(pts, 'shape') and len(pts.shape) == 2 else 1)
        # Using .view so we figure out N dynamically
        if len(pts.shape) == 2:
            Bs = point_out.batch.max().item() + 1
            N = pts.shape[0] // Bs
        else:
            Bs, N, _ = pts.shape
            
        pts_feat_from_voxel = point_out.feat      # (V, C_out)
        pts_coord_from_voxel = point_out.coord    # (V, 3)
        pts_batch_from_voxel = point_out.batch    # (V,)

        original_pts = pts.view(Bs, N, 3)         # (Bs, N, 3)

        voxel_coords_split = [pts_coord_from_voxel[pts_batch_from_voxel == b] for b in range(Bs)]
        voxel_feats_split  = [pts_feat_from_voxel[pts_batch_from_voxel == b] for b in range(Bs)]
        voxel_lens         = torch.tensor([v.shape[0] for v in voxel_coords_split], device=pts.device)

        max_T = int(voxel_lens.max().item())
        pad_3 = (0, 0, 0, max_T)
        pad_C = (0, 0, 0, max_T)

        voxel_coords_padded = torch.stack(
            [F.pad(v, pad_3, value=0.)[:max_T] for v in voxel_coords_split]
        )  # (Bs, max_T, 3)
        voxel_feats_padded  = torch.stack(
            [F.pad(v, pad_C, value=0.)[:max_T] for v in voxel_feats_split]
        )  # (Bs, max_T, C_out)

        # ------------------------------------------------------------------
        # Batched KNN -------------------------------------------------------
        # ------------------------------------------------------------------
        K_interp = self.k_neighbors

        knn = knn_points(
            original_pts,
            voxel_coords_padded,
            lengths2=voxel_lens,
            K=K_interp,
        )
        dists = knn.dists  # (Bs, N, K)
        idx   = knn.idx    # (Bs, N, K)
        
        interpolated_feats_chunks = []
        num_chunks = (N + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, N)

            # Slice the indices and distances for the current chunk
            idx_chunk = idx[:, start_idx:end_idx, :]      # (Bs, chunk_size, K)
            dists_chunk = dists[:, start_idx:end_idx, :]  # (Bs, chunk_size, K)

            # Gather features only for the neighbors of the current chunk's points
            gathered_feats_chunk = knn_gather(voxel_feats_padded, idx_chunk) # (Bs, chunk_size, K, C_out)

            # Calculate inverse distance weights for the chunk
            weights_chunk = 1.0 / (dists_chunk + 1e-8)
            weights_chunk = weights_chunk / weights_chunk.sum(dim=-1, keepdim=True) # (Bs, chunk_size, K)

            # Perform the weighted sum for the chunk
            interpolated_chunk = (gathered_feats_chunk * weights_chunk.unsqueeze(-1)).sum(dim=-2) # (Bs, chunk_size, C_out)
            interpolated_feats_chunks.append(interpolated_chunk)

        # Concatenate the processed chunks back into a single tensor
        interpolated_feats = torch.cat(interpolated_feats_chunks, dim=1)  # (Bs, N, C_out)
        return interpolated_feats


    def forward(self, pts, feats, voxel_sizes, chunk_size=50000):
        '''
        Forward pass for the back-end processing.
        
        Params:
            - pts: (Bs, N, 3) tensor of point coordinates
            - feats: (Bs, N, C) tensor of point features
            - voxel_sizes: list of voxel sizes
            - chunk_size: int, number of points to process in each chunk for interpolation
        '''

        Bs, C = feats.shape[0], feats.shape[-1]

        if len(feats.shape) != 3:
            feats = feats.reshape(Bs, -1, C)
            pts = pts.reshape(Bs, -1, 3) 
            
        
        bounding_boxes = torch.zeros((Bs, 2, 3), device=pts.device)  # Dummy bounding boxes
        bounding_boxes[:, 0, :] = pts.view(Bs, -1, 3).min(dim=1).values
        bounding_boxes[:, 1, :] = pts.view(Bs, -1, 3).max(dim=1).values # Bs, 2, 3

        Bs, N, C = feats.shape
        pts = pts.reshape(-1, 3)
        feats = feats.reshape(-1, C)
        batch_ids = torch.arange(Bs).repeat_interleave(N).to(pts.device)

        level_feats = []

        for i, voxel_size in enumerate(voxel_sizes):
            feat, info = self.mean_by_voxel(pts, feats, batch_ids, voxel_size, bounding_boxes)
            vox_id = info['unique_indices']

            data_dict = {
                'feat': feat,
                'grid_coord': vox_id[:, 1:],
                'coord': voxel_size[vox_id[:, 0]] * vox_id[:, 1:] if isinstance(voxel_size, torch.Tensor) else voxel_size * vox_id[:, 1:], # FIXME
                'batch': vox_id[:, 0],
            }

            point_out = self.point_transformer(data_dict)
            interpolated_feats = self.voxel_to_point_interpolation(
                point_out, pts, chunk_size
            )
            level_feats.append(interpolated_feats)
        
        return level_feats
