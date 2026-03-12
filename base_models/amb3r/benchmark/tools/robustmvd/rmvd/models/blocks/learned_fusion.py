# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/models/blocks/learned_fusion.py

import torch
import torch.nn as nn


class LearnedFusion(nn.Module):
    def __init__(self):

        super().__init__()

        self.view_weights = []
        self.fused_corr = None
        self.fused_mask = None

        self.corr_to_view_weight = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def reset(self):
        self.view_weights = []
        self.fused_corr = None
        self.fused_mask = None

    def forward(self, corrs, masks):
        self.reset()

        if len(corrs) > 1:
            for corr, mask in zip(corrs, masks):
                view_weight = self.corr_to_view_weight(corr)
                self.view_weights.append(view_weight)

            self.view_weights = torch.stack(self.view_weights, dim=0)
            self.view_weights = nn.functional.softmax(self.view_weights, dim=0) + 1e-9
            self.view_weights = torch.split(self.view_weights, [1] * len(corrs), dim=0)
            self.view_weights = [x[0] for x in self.view_weights]

            view_weights = [view_weight * mask for view_weight, mask in zip(self.view_weights, masks)]  # each NSHW
            view_weights_sum = torch.stack(view_weights, dim=0).sum(0)  # N, S, H, W
            self.fused_mask = (view_weights_sum != 0).float()

            corr_sum = (torch.stack(corrs, dim=0) * torch.stack(view_weights, dim=0)).sum(0)
            corr_avg = corr_sum / (view_weights_sum + 1e-9) * self.fused_mask
            self.fused_corr = corr_avg

        else:
            self.fused_corr = corrs[0]
            self.fused_mask = masks[0]

        return self.fused_corr, self.fused_mask
