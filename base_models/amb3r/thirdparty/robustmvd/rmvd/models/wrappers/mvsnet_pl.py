# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/models/wrappers/mvsnet_pl.py

import os.path as osp
import math

import torch
from torchvision import transforms as T
import torch.nn as nn
import numpy as np

from ..registry import register_model
from ..helpers import build_model_with_cfg
from rmvd.utils import get_path, get_torch_model_device, to_numpy, to_torch, select_by_index, exclude_index
from rmvd.data.transforms import ResizeInputs


class MVSNet_pl_Wrapped(nn.Module):
    def __init__(self, sample_in_inv_depth_space=False, num_sampling_steps=192):
        super().__init__()

        import sys
        paths_file = osp.join(osp.dirname(osp.realpath(__file__)), 'paths.toml')
        repo_path = get_path(paths_file, "mvsnet_pl", "root")
        sys.path.insert(0, repo_path)
        from models.mvsnet import MVSNet
        self.model = MVSNet()

        weights_path = osp.join(repo_path, "_ckpt_epoch_14.ckpt")
        print(f"Using model weights from {weights_path}.")
        weights = torch.load(weights_path)['state_dict']
        fixed_weights = {}
        for k, v in weights.items():
            fixed_weights[k[6:]] = v
        self.model.load_state_dict(fixed_weights)
        
        self.sample_in_inv_depth_space = sample_in_inv_depth_space
        self.num_sampling_steps = num_sampling_steps

        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def input_adapter(self, images, keyview_idx, poses=None, intrinsics=None, depth_range=None):
        device = get_torch_model_device(self)

        N = images[0].shape[0]
        orig_ht, orig_wd = images[0].shape[-2:]
        ht, wd = int(math.ceil(orig_ht / 64.0) * 64.0), int(math.ceil(orig_wd / 64.0) * 64.0)
        if (orig_ht != ht) or (orig_wd != wd):
            resized = ResizeInputs(size=(ht, wd))({'images': images, 'intrinsics': intrinsics})
            images = resized['images']
            intrinsics = resized['intrinsics']

        for idx, image_batch in enumerate(images):
            tmp_images = []
            image_batch = image_batch.transpose(0, 2, 3, 1)
            for image in image_batch:
                image = self.input_transform(image.astype(np.uint8)).float()
                tmp_images.append(image)

            image_batch = torch.stack(tmp_images)
            images[idx] = image_batch

        proj_mats = []
        for idx, (intrinsic_batch, pose_batch) in enumerate(zip(intrinsics, poses)):
            proj_mat_batch = []
            for intrinsic, pose, cur_keyview_idx in zip(intrinsic_batch, pose_batch, keyview_idx):

                scale_arr = np.array([[0.25] * 3, [0.25] * 3, [1.] * 3])  # 3, 3
                intrinsic = intrinsic * scale_arr  # scale intrinsics to 4x downsampling that happens within the model

                proj_mat = pose
                proj_mat[:3, :4] = intrinsic @ proj_mat[:3, :4]
                proj_mat = proj_mat.astype(np.float32)

                if idx == cur_keyview_idx:
                    proj_mat = np.linalg.inv(proj_mat)

                proj_mat_batch.append(proj_mat)

            proj_mat_batch = np.stack(proj_mat_batch)
            proj_mats.append(proj_mat_batch)
        
        if depth_range is None:
            if self.sample_in_inv_depth_space:
                depth_samples = 1 / np.linspace(1 / 100, 1 / 0.2, self.num_sampling_steps, dtype=np.float32)[::-1]
            else:
                depth_samples = np.linspace(0.2, 100, self.num_sampling_steps, dtype=np.float32)

            depth_samples = np.stack(N*[depth_samples])
        else:
            min_depth, max_depth = depth_range
            if self.sample_in_inv_depth_space:
                depth_samples = 1 / np.linspace(1 / max_depth, 1 / min_depth, self.num_sampling_steps,
                                                dtype=np.float32)[::-1]
            else:
                depth_samples = np.linspace(min_depth, max_depth, self.num_sampling_steps, dtype=np.float32)
            depth_samples = depth_samples.transpose()  # (num_sampling_steps, N) to (N, num_sampling_steps)

        images, keyview_idx, proj_mats, depth_samples = \
            to_torch((images, keyview_idx, proj_mats, depth_samples), device=device)

        sample = {
            'images': images,
            'keyview_idx': keyview_idx,
            'proj_mats': proj_mats,
            'depth_samples': depth_samples,
        }
        return sample

    def forward(self, images, proj_mats, depth_samples, keyview_idx, **_):

        # TODO: move this to input_adapter
        image_key = select_by_index(images, keyview_idx)
        images_source = exclude_index(images, keyview_idx)
        images = [image_key] + images_source

        proj_mat_key = select_by_index(proj_mats, keyview_idx)
        proj_mats_source = exclude_index(proj_mats, keyview_idx)
        proj_mats = [proj_mat_key] + proj_mats_source

        images = torch.stack(images, 1)  # N, num_views, 3, H, W
        proj_mats = torch.stack(proj_mats, 1)  # N, num_views, 4, 4

        pred_depth, pred_depth_confidence = self.model.forward(images, proj_mats, depth_samples)
        pred_depth_uncertainty = 1 - pred_depth_confidence

        pred_depth = pred_depth.unsqueeze(1)
        pred_depth_uncertainty = pred_depth_uncertainty.unsqueeze(1)

        pred = {
            'depth': pred_depth,
            'depth_uncertainty': pred_depth_uncertainty
        }
        aux = {}

        return pred, aux

    def output_adapter(self, model_output):
        pred, aux = model_output
        return to_numpy(pred), to_numpy(aux)


@register_model(trainable=False)
def mvsnet_pl_wrapped(pretrained=True, weights=None, train=False, num_gpus=1, **kwargs):
    assert pretrained and (weights is None), "Model supports only pretrained=True, weights=None."
    cfg = {"sample_in_inv_depth_space": False, "num_sampling_steps": 192}
    model = build_model_with_cfg(model_cls=MVSNet_pl_Wrapped, cfg=cfg, weights=None, train=train, num_gpus=num_gpus)
    return model
