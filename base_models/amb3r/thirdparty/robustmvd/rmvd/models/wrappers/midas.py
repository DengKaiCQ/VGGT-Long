# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/models/wrappers/midas.py

import os.path as osp

import torch
import torch.nn as nn
from torchvision.transforms import Compose
import numpy as np
import cv2

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import get_path, get_torch_model_device, to_numpy, to_torch, select_by_index


class MiDaS_Wrapped(nn.Module):
    def __init__(self, weights_name):
        super().__init__()

        import sys
        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        repo_path = get_path(paths_file, "midas", "root")
        sys.path.insert(0, repo_path)
        from midas.midas_net import MidasNet
        from midas.transforms import Resize, NormalizeImage, PrepareForNet

        weights_path = osp.join(repo_path, "weights", weights_name)
        self.model = MidasNet(weights_path, non_negative=True)

        net_w, net_h = 384, 384
        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        image_batch = select_by_index(images, keyview_idx)
        tmp_images = []
        for image in image_batch:
            image = image / 255.
            image = np.transpose(image, [1, 2, 0])
            image = self.transform({"image": image})["image"]
            tmp_images.append(image)
        image = np.stack(tmp_images)

        image = to_torch(image, device=device)

        sample = {'image': image}
        return sample

    def forward(self, image, **_):
        pred_invdepth = self.model(image)
        return pred_invdepth

    def output_adapter(self, model_output):
        pred_invdepth = model_output
        pred_invdepth = to_numpy(pred_invdepth)
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_depth = 1 / pred_invdepth
        pred_depth = pred_depth[:, None]
        pred = {'depth': pred_depth}
        aux = {}
        return pred, aux


@register_model(trainable=False)
def midas_big_v2_1_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {"weights_name": "midas_v21-f6b98070.pt"}
    model = build_model_with_cfg(model_cls=MiDaS_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model
