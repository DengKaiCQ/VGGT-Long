# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/models/robust_mvd.py

import math

import torch
import torch.nn as nn
import numpy as np

from .registry import register_model
from .helpers import build_model_with_cfg
from .blocks.dispnet_context_encoder import DispnetContextEncoder
from .blocks.dispnet_encoder import DispnetEncoder
from .blocks.planesweep_corr import PlanesweepCorrelation
from .blocks.learned_fusion import LearnedFusion
from .blocks.dispnet_costvolume_encoder import DispnetCostvolumeEncoder
from .blocks.dispnet_decoder import DispnetDecoder

from rmvd.utils import get_torch_model_device, to_numpy, to_torch, select_by_index, exclude_index
from rmvd.data.transforms import ResizeInputs


class RobustMVD(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = DispnetEncoder()
        self.context_encoder = DispnetContextEncoder()
        self.corr_block = PlanesweepCorrelation()
        self.fusion_block = LearnedFusion()
        self.fusion_enc_block = DispnetCostvolumeEncoder()
        self.decoder = DispnetDecoder()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv3d) or isinstance(
                    m, nn.ConvTranspose3d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, poses, intrinsics, keyview_idx, **_):

        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)

        intrinsics_key = select_by_index(intrinsics, keyview_idx)
        intrinsics_source = exclude_index(intrinsics, keyview_idx)

        source_to_key_transforms = exclude_index(poses, keyview_idx)

        all_enc_key, enc_key = self.encoder(image_key)
        enc_sources = [self.encoder(image_source)[1] for image_source in images_source]

        ctx = self.context_encoder(enc_key)

        corrs, masks = self.corr_block(feat_key=enc_key, intrinsics_key=intrinsics_key, feat_sources=enc_sources,
                                       source_to_key_transforms=source_to_key_transforms,
                                       intrinsics_sources=intrinsics_source)

        fused_corr, fused_mask = self.fusion_block(corrs=corrs, masks=masks)

        all_enc_fused, enc_fused = self.fusion_enc_block(corr=fused_corr, ctx=ctx)

        dec = self.decoder(enc_fused=enc_fused, all_enc={**all_enc_key, **all_enc_fused})

        pred = {
            'depth': 1 / (dec['invdepth'] + 1e-9),
            'depth_uncertainty': torch.exp(dec['invdepth_log_b']) / (dec['invdepth'] + 1e-9)
        }
        aux = dec

        return pred, aux

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        orig_ht, orig_wd = images[0].shape[-2:]
        ht, wd = int(math.ceil(orig_ht / 64.0) * 64.0), int(math.ceil(orig_wd / 64.0) * 64.0)
        if (orig_ht != ht) or (orig_wd != wd):
            resized = ResizeInputs(size=(ht, wd))({'images': images, 'intrinsics': intrinsics})
            images = resized['images']
            intrinsics = resized['intrinsics']

        # normalize images
        images = [image / 255.0 - 0.4 for image in images]

        # model works with relative intrinsics:
        scale_arr = np.array([[wd]*3, [ht]*3, [1.]*3], dtype=np.float32)  # 3, 3
        intrinsics = [intrinsic / scale_arr for intrinsic in intrinsics]

        images, keyview_idx, poses, intrinsics, depth_range = \
            to_torch((images, keyview_idx, poses, intrinsics, depth_range), device=device)

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
            'poses': poses,
            'intrinsics': intrinsics,
            'depth_range': depth_range,
        }
        return sample

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def robust_mvd_5M(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    pretrained_weights = 'https://lmb.informatik.uni-freiburg.de/people/schroepp/weights/robustmvd.pt'
    weights = pretrained_weights if (pretrained and weights is None) else None
    model = build_model_with_cfg(model_cls=RobustMVD, weights=weights, train=train, num_gpus=num_gpus)
    return model


@register_model
def robust_mvd(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    pretrained_weights = 'https://lmb.informatik.uni-freiburg.de/people/schroepp/weights/robustmvd_600k.pt'
    weights = pretrained_weights if (pretrained and weights is None) else None
    model = build_model_with_cfg(model_cls=RobustMVD, weights=weights, train=train, num_gpus=num_gpus)
    return model
