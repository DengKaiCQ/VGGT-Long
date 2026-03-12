# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/loss/multi_scale_uni_laplace.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import m_univariate_laplace_nll, pointwise_univariate_laplace_nll, mae, pointwise_ae
from .registry import register_loss


class MultiScaleUniLaplace(nn.Module):
    def __init__(self, model, weight_decay=1e-4, gt_interpolation="nearest", modality="invdepth", verbose=True,
                 deterministic_loss_iterations=2000, mean_scaling_factor=1):

        super().__init__()

        self.verbose = verbose

        if self.verbose:
            print(f"Initializing {self.name} loss.")

        self.weight_decay = weight_decay
        self.gt_interpolation = gt_interpolation

        self.loss_weights = [1 / 16, 1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]
        self.loss_weights = [100 * weight for weight in self.loss_weights]

        self.modality = modality
        self.deterministic_loss_iterations = deterministic_loss_iterations
        self.mean_scaling_factor = mean_scaling_factor

        self.reg_params = self.get_regularization_parameters(model)  # TODO: I think there is a better way in pytorch to do this

        if self.verbose:
            print(f"\tWeight decay: {self.weight_decay}")
            print(f"\tGT interpolation: {self.gt_interpolation}")
            print(f"\tModality: {self.modality}")
            print(f"\tLoss weights: {self.loss_weights}")
            print(f"Finished initializing {self.name} loss.")
            print()

    @property
    def name(self):
        name = type(self).__name__
        return name

    def get_regularization_parameters(self, model):
        reg_params = []
        for name, param in model.named_parameters():
            if "pred" not in name and not name.endswith("bias") and not name.endswith(
                    "bn.weight") and param.requires_grad:
                reg_params.append((name, param))

        if self.verbose:
            print(f"\tApplying regularization loss with weight decay {self.weight_decay} on:")
            for i, val in enumerate(reg_params):
                name, param = val
                print(f"\t\t#{i} {name}: {param.shape} ({param.numel()})")

        return reg_params

    def forward(self, sample_inputs, sample_gt, pred, aux, iteration):

        sub_losses = {}
        pointwise_losses = {}

        gt = sample_gt[self.modality] * self.mean_scaling_factor
        gt_mask = gt > 0

        preds_all = [x * self.mean_scaling_factor for x in aux[f"{self.modality}s_all"]]
        pred_log_bs_all = aux[f"{self.modality}_log_bs_all"]

        total_mnll_loss = 0
        total_reg_loss = 0

        for level, (pred, pred_log_b) in enumerate(zip(preds_all, pred_log_bs_all)):

            with torch.no_grad():
                gt_resampled = F.interpolate(gt, size=pred.shape[-2:], mode=self.gt_interpolation)
                gt_mask_resampled = F.interpolate(gt_mask.float(), size=pred.shape[-2:], mode="nearest") == 1.0

            if iteration < self.deterministic_loss_iterations:
                loss = mae(gt=gt_resampled, pred=pred, mask=gt_mask_resampled, weight=self.loss_weights[level])
                pointwise_loss = pointwise_ae(gt=gt_resampled, pred=pred, mask=gt_mask_resampled,
                                              weight=self.loss_weights[level])

            else:
                loss = m_univariate_laplace_nll(gt=gt_resampled, pred_a=pred, pred_log_b=pred_log_b,
                                                mask=gt_mask_resampled, weight=self.loss_weights[level])
                pointwise_loss = pointwise_univariate_laplace_nll(gt=gt_resampled, pred_a=pred, pred_log_b=pred_log_b,
                                                                  mask=gt_mask_resampled,
                                                                  weight=self.loss_weights[level])

            sub_losses["02_mnll/level_%d" % level] = loss
            pointwise_losses["00_nll/level_%d" % level] = pointwise_loss

            total_mnll_loss += loss

        for name, param in self.reg_params:
            reg_loss = torch.sum(torch.mul(param, param)) / 2.0
            total_reg_loss += reg_loss
        total_reg_loss *= self.weight_decay

        total_loss = total_mnll_loss + total_reg_loss

        sub_losses["00_total_mnll"] = total_mnll_loss
        sub_losses["01_reg"] = total_reg_loss
        return total_loss, sub_losses, pointwise_losses


@register_loss
def robust_mvd_loss(**kwargs):
    return MultiScaleUniLaplace(weight_decay=1e-4, gt_interpolation="nearest", modality="invdepth", deterministic_loss_iterations=2000, 
                                mean_scaling_factor=1050, **kwargs)
