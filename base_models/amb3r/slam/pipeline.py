# References:
#   https://github.com/HengyiWang/amb3r/blob/main/slam/pipeline.py

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

from omegaconf import OmegaConf
from slam.memory import SLAMemory
from benchmark.tools.pose_eval import evaluate_evo


class AMB3R_VO():
    def __init__(self, model, cfg_path='./slam/slam_config.yaml'):
        self.cfg = OmegaConf.load(cfg_path)
        self.model = model.to(self.cfg.device)
    

    @torch.no_grad()
    def local_mapping(self, views_all, cfg, init=False):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            res = self.model.run_amb3r_vo(views_all, cfg, self.keyframe_memory if not init else None)
        return res
    
    def initialize_map(self, views_all, cfg):
        res = self.local_mapping(views_all, cfg, init=True)
        conf = res['world_points_conf']
        avg_confidence = conf[0].mean(dim=(-1, -2)).cpu().numpy()
        print("Average confidence values:", [f"{c:.2f}" for c in avg_confidence])
        self.keyframe_memory.initialize(res)               
    
    def mapping(self, views_all, cfg):
        res = self.local_mapping(views_all, cfg)
        self.keyframe_memory.update(res, start_idx=views_all['start_idx'], end_idx=views_all['end_idx'])
        
        
    def run(self, images, poses_gt=None):
        '''
        Params:
            - images: (B, T, 3, H, W), in [-1, 1] range
            - poses_gt: (B, T, 4, 4), in world space
        '''

        assert images.min() >= -1 and images.max() <= 1, "Images should be in [-1, 1] range"

        Bs, T, _, H, W = images.shape

        initialized = False
        self.keyframe_memory = SLAMemory(self.cfg, T, H, W)


        # Let's SLAM
        for idx in range(T):
            if idx < self.cfg.map_init_window - 1:
                # Wait until we have enough frames
                continue

            if not initialized:
                # Map initialization
                views_all = {}
                views_all['images'] = images[:, :idx+1].to(self.cfg.device)
                self.initialize_map(views_all, self.cfg)
                initialized = True

                last_mapped_idx = idx   
            
            else:
                if (idx - last_mapped_idx < self.cfg.map_every) and (idx < T - 1):
                    continue

                start_idx = idx+1- self.cfg.map_every

                views_to_map = {
                    'images': torch.cat([images[:, self.keyframe_memory.cur_kf_idx], images[:, start_idx:idx+1]], dim=1).to(self.cfg.device),
                    'start_idx': start_idx,
                    'end_idx': idx
                }

                self.mapping(views_to_map, self.cfg)
                last_mapped_idx = idx
            
            print(f"Processed frame {idx+1}/{T}, KF ids: {self.keyframe_memory.cur_kf_idx.tolist()}")
            try:
                evaluate_evo(poses_gt[:idx+1], self.keyframe_memory.poses[:idx+1].numpy(), None, None, monocular=True, plot=False) 
            except:
                pass

        return self.keyframe_memory


                



                
                
            







        