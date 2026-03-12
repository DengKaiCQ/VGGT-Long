# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/models/blocks/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    mod = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                  bias=True),
        nn.LeakyReLU(0.2, inplace=True)
    )

    return mod


class ReLUAndSigmoid(nn.Module):

    def __init__(self, inplace: bool = False, min: float = 0, max: float = 1):
        super(ReLUAndSigmoid, self).__init__()
        self.inplace = inplace
        self.min = min
        self.max = max
        self.range = max - min

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input0 = F.relu(input[:, :1, :, :], inplace=self.inplace)
        input1 = (torch.sigmoid(input[:, 1:, :, :] * (4 / self.range)) * self.range) + self.min
        return torch.cat([input0, input1], 1)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
