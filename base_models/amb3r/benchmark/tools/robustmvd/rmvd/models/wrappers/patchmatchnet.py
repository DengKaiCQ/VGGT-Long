# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/models/wrappers/patchmatchnet.py

import os.path as osp
import math

import torch
import torch.nn as nn
from torchvision import transforms as T
import numpy as np
from easydict import EasyDict

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import (
    get_path,
    get_torch_model_device,
    to_numpy,
    to_torch,
    select_by_index,
    exclude_index,
)


class PatchmatchNet_Wrapped(nn.Module):
    def __init__(self, num_sampling_steps=192):
        super().__init__()

        import sys

        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), "paths.toml")
        repo_path = get_path(paths_file, "patchmatchnet", "root")
        sys.path.insert(0, repo_path)

        from models.net import PatchmatchNet

        patchmatch_interval_scale = [0.005, 0.0125, 0.025]
        patchmatch_range = [6, 4, 2]
        patchmatch_iteration = [1, 2, 2]
        patchmatch_num_sample = [8, 8, 16]
        propagate_neighbors = [0, 8, 16]
        evaluate_neighbors = [9, 9, 9]
        self.model = PatchmatchNet(
            patchmatch_interval_scale=patchmatch_interval_scale,
            propagation_range=patchmatch_range,
            patchmatch_iteration=patchmatch_iteration,
            patchmatch_num_sample=patchmatch_num_sample,
            propagate_neighbors=propagate_neighbors,
            evaluate_neighbors=evaluate_neighbors
        )
        state_dict = torch.load(osp.join(repo_path, "checkpoints/params_000007.ckpt"))["model"]
        fixed_weights = {}
        for k, v in state_dict.items():
            fixed_weights[k[7:]] = v
        self.model.load_state_dict(fixed_weights)

        self.num_sampling_steps = num_sampling_steps

    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None  # TODO: does it make sense that poses etc are set to None?
    ):
        device = get_torch_model_device(self)

        # normalize images
        images = [image / 255.0 for image in images]

        depth_range = [np.array([0.2], dtype=np.float32), np.array([100], dtype=np.float32)] if depth_range is None else depth_range
        min_depth, max_depth = depth_range

        images, keyview_idx, poses, intrinsics, min_depth, max_depth = to_torch(
            (images, keyview_idx, poses, intrinsics, min_depth, max_depth), device=device)

        # TODO: check min_depth, max_depth dtype with given or default depth range

        sample = {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "keyview_idx": keyview_idx,
            "min_depth": min_depth,
            "max_depth": max_depth,
        }
        return sample

    def forward(self, images,  poses, intrinsics, keyview_idx, min_depth, max_depth, **_):
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        images = [image_key] + images_source

        intrinsics_key = select_by_index(intrinsics, keyview_idx)  # N, 3, 3
        intrinsics_source = exclude_index(intrinsics, keyview_idx)
        intrinsics = [intrinsics_key] + intrinsics_source
        intrinsics = torch.stack(intrinsics, dim=1)  # N, NV, 3, 3

        pose_key = select_by_index(poses, keyview_idx)  # N, 4, 4
        poses_source = exclude_index(poses, keyview_idx)
        poses = [pose_key] + poses_source
        poses = torch.stack(poses, dim=1)  # N, NV, 4, 4

        pred_depth, pred_depth_confidence, _ = self.model.forward(images, intrinsics, poses, min_depth, max_depth)
        # N, 1, H, W and N, H, W
        pred_depth_confidence = pred_depth_confidence.unsqueeze(1)
        pred_depth_uncertainty = 1 - pred_depth_confidence

        pred = {"depth": pred_depth, "depth_uncertainty": pred_depth_uncertainty}
        aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def patchmatchnet_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    cfg = {
        "num_sampling_steps": 192,
    }

    model = build_model_with_cfg(
        model_cls=PatchmatchNet_Wrapped,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model
