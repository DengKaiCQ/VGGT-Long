# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/models/blocks/dispnet_context_encoder.py

import torch.nn as nn

from .utils import conv


class DispnetContextEncoder(nn.Module):
    def __init__(self):

        super().__init__()
        self.conv_redir = conv(256, 32, kernel_size=1, stride=1)

    def forward(self, conv3):
        conv_redir = self.conv_redir(conv3)
        return conv_redir
