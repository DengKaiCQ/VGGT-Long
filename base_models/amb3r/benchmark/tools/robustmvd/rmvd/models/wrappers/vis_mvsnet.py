# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/models/wrappers/vis_mvsnet.py

import os.path as osp
import math

import torch
import torch.nn as nn
from torchvision import transforms as T
import numpy as np

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


class VisMVSNet_Wrapped(nn.Module):
    def __init__(self, num_sampling_steps=192):
        super().__init__()

        import sys

        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), "paths.toml")
        repo_path = get_path(paths_file, "vis_mvsnet", "root")
        sys.path.insert(0, repo_path)

        from core.model_cas import Model

        self.model = Model()
        state_dict = torch.load(osp.join(repo_path, "pretrained_model/vis/20000.tar"))[
            "state_dict"
        ]
        fixed_weights = {}
        for k, v in state_dict.items():
            fixed_weights[k[7:]] = v
        self.model.load_state_dict(fixed_weights)

        self.input_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.num_sampling_steps = num_sampling_steps

    def input_adapter(
        self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None
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

        for idx, image_batch in enumerate(images):
            tmp_images = []
            image_batch = image_batch.transpose(0, 2, 3, 1)
            for image in image_batch:
                image = self.input_transform(image.astype(np.uint8)).float()
                image = torch.flip(image, [0])  # RGB to BGR
                tmp_images.append(image)

            image_batch = torch.stack(tmp_images)
            images[idx] = image_batch

        depth_range = [0.2, 100] if depth_range is None else depth_range
        min_depth, max_depth = depth_range
        step_size = (max_depth - min_depth) / self.num_sampling_steps

        cams = []
        for idx, (intrinsic, pose) in enumerate(zip(intrinsics, poses)):

            cam = np.zeros((2, 4, 4), dtype=np.float32)
            cam[0] = pose
            cam[1, :3, :3] = intrinsic
            cam[1, 3, 0] = min_depth
            cam[1, 3, 1] = step_size
            cam[1, 3, 2] = self.num_sampling_steps
            cam[1, 3, 3] = max_depth

            cam = cam[np.newaxis, :]  # 1, 2, 4, 4
            cams.append(cam)

        images, keyview_idx, cams = to_torch((images, keyview_idx, cams), device=device)

        sample = {
            "images": images,
            "keyview_idx": keyview_idx,
            "cams": cams,
        }
        return sample

    def forward(self, images, cams, keyview_idx, **_):
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)

        cam_key = select_by_index(cams, keyview_idx)
        cam_source = exclude_index(cams, keyview_idx)

        images_source = torch.stack(images_source, 1)  # N, num_views, 3, H, W
        cam_source = torch.stack(cam_source, 1)  # N, num_views, 4, 4

        inp = {
            "ref": image_key,
            "ref_cam": cam_key,
            "srcs": images_source,
            "srcs_cam": cam_source,
        }

        cas_depth_num = [64, 32, 16]
        cas_interv_scale = [4.0, 2.0, 1.0]
        outputs, refined_depth, prob_maps = self.model(
            inp, cas_depth_num, cas_interv_scale, mode="soft"
        )
        pred_depth = refined_depth
        pred_depth_confidence = prob_maps[2]
        pred_depth_uncertainty = 1 - pred_depth_confidence

        pred = {"depth": pred_depth, "depth_uncertainty": pred_depth_uncertainty}
        aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def vis_mvsnet_wrapped(
    pretrained=True, weights=None, train=False, num_gpus=1, **kwargs
):
    assert pretrained and (
        weights is None
    ), "Model supports only pretrained=True, weights=None."

    cfg = {
        "num_sampling_steps": 192,
    }

    model = build_model_with_cfg(
        model_cls=VisMVSNet_Wrapped,
        cfg=cfg,
        weights=None,
        train=train,
        num_gpus=num_gpus,
    )
    return model
