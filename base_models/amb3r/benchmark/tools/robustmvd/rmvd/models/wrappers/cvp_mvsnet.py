# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/models/wrappers/cvp_mvsnet.py

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
from rmvd.data.transforms import ResizeInputs


class CVPMVSNet_Wrapped(nn.Module):
    def __init__(self, num_sampling_steps=192):
        super().__init__()

        import sys

        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), "paths.toml")
        repo_path = get_path(paths_file, "cvp_mvsnet", "root")
        sys.path.insert(0, osp.join(repo_path, "CVP_MVSNet"))

        from models.net import network

        self.args = EasyDict({  # parameters are taken from original repository when executing eval.sh script
            'nsrc': None,  # will be set in forward()
            'nscale': 5,
            'mode': 'test',
        })
        self.model = network(self.args)
        state_dict = torch.load(osp.join(repo_path, "CVP_MVSNet/checkpoints/pretrained/model_000027.ckpt"))[
            "model"
        ]
        self.model.load_state_dict(state_dict, strict=False)

        self.num_sampling_steps = num_sampling_steps

    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None  # TODO: does it make sense that poses etc are set to None?
    ):
        device = get_torch_model_device(self)

        orig_ht, orig_wd = images[0].shape[-2:]
        ht, wd = int(math.ceil(orig_ht / 64.0) * 64.0), int(
            math.ceil(orig_wd / 64.0) * 64.0
        )
        if (orig_ht != ht) or (orig_wd != wd):
            resized = ResizeInputs(size=(ht, wd))(
                {"images": images, "intrinsics": intrinsics}
            )
            images = resized["images"]
            intrinsics = resized["intrinsics"]

        # normalize images
        images = [image / 255.0 for image in images]

        depth_range = [np.array([0.2]), np.array([100])] if depth_range is None else depth_range
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
        self.args.nsrc = len(images_source)
        images_source = torch.stack(images_source, dim=1)  # N, NV, 3, H, W

        intrinsics_key = select_by_index(intrinsics, keyview_idx)  # N, 3, 3
        intrinsics_source = exclude_index(intrinsics, keyview_idx)  # N, NV, 3, 3
        intrinsics_source = torch.stack(intrinsics_source, dim=1)  # N, NV, 3, 3

        pose_key = select_by_index(poses, keyview_idx)  # N, 4, 4
        poses_source = exclude_index(poses, keyview_idx)
        poses_source = torch.stack(poses_source, dim=1)  # N, NV, 4, 4

        inp = {
            "ref_img": image_key,
            "src_imgs": images_source,
            "ref_in": intrinsics_key,
            "src_in": intrinsics_source,
            "ref_ex": pose_key,
            "src_ex": poses_source,
            "depth_min": min_depth,
            "depth_max": max_depth,
        }

        outputs = self.model(**inp)
        pred_depth = outputs["depth_est_list"][0]  # N, H, W
        pred_depth_confidence = outputs["prob_confidence"]
        pred_depth_uncertainty = 1 - pred_depth_confidence  # N, H, W

        pred_depth = pred_depth.unsqueeze(1)
        pred_depth_uncertainty = pred_depth_uncertainty.unsqueeze(1)

        pred = {"depth": pred_depth, "depth_uncertainty": pred_depth_uncertainty}
        aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def cvp_mvsnet_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    cfg = {
        "num_sampling_steps": 192,
    }

    model = build_model_with_cfg(
        model_cls=CVPMVSNet_Wrapped,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model
