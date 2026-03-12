# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/models/blocks/dispnet_costvolume_encoder.py

import torch
import torch.nn as nn

from .utils import conv


class DispnetCostvolumeEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        C_curr = 256
        self.conv3_1 = conv(256 + 32, C_curr)

        C_last = C_curr  # 256
        C_curr *= 2  # 512
        self.conv4 = conv(C_last, C_curr, stride=2)
        self.conv4_1 = conv(C_curr, C_curr)
        self.conv5 = conv(C_curr, C_curr, stride=2)
        self.conv5_1 = conv(C_curr, C_curr)

        C_last = C_curr  # 512
        C_curr *= 2  # 1024
        self.conv6 = conv(C_last, C_curr, stride=2)
        self.conv6_1 = conv(C_curr, C_curr)

    def forward(self, corr, ctx):

        merged = torch.cat([ctx, corr], 1)
        conv3_1 = self.conv3_1(merged)

        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)

        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)

        conv6 = self.conv6(conv5_1)
        conv6_1 = self.conv6_1(conv6)

        all_enc = {
            'merged': merged,
            'conv3_1': conv3_1,
            'conv4': conv4,
            'conv4_1': conv4_1,
            'conv5': conv5,
            'conv5_1': conv5_1,
            'conv6': conv6,
            'conv6_1': conv6_1,
        }

        return all_enc, conv6_1
