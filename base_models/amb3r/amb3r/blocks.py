# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/blocks.py

import torch
import torch.nn as nn


class ZeroConvBlock(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=1024, out_channels=1024, enable_zero_conv=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.enable_zero_conv = enable_zero_conv

        if enable_zero_conv:
            nn.init.constant_(self.conv2.weight, 0.0)
            nn.init.constant_(self.conv2.bias, 0.0)
        

    def forward(self, x):
        if self.enable_zero_conv:
            return self.conv2(self.act(self.conv1(x)))
        return self.conv2(self.act(self.conv1(x))) + x


class DownBlock(nn.Module):
    def __init__(self, in_channels=1024, mid_channels=1024, out_channels=1024):
        super().__init__()
        mid_channels = mid_channels or in_channels
        out_channels = out_channels or in_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, mid_channels),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, mid_channels), 

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, mid_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(32, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class ScaleProjector(nn.Module):
    def __init__(self, depth_feat_channels=128, token_dim=1024, cnn_output_dim=512):
        super().__init__()
        
        self.depth_cnn_downsampler = nn.Sequential(
            nn.Conv2d(depth_feat_channels, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling
        )
        
        self.token_aggregator = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.LayerNorm(token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim//2)
        )

        self.combined_projector = nn.Sequential(
            nn.Linear(cnn_output_dim + token_dim // 2, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, depth_feat, enc_tokens):
        """
        Forward pass for the fusion and scale prediction.
        
        Args:
            depth_feat (torch.Tensor): The depth feature map. 
                                       Assumed Shape: (Bs, nimgs, Channels, H, W)
            enc_tokens (torch.Tensor): The tokens from a transformer.
                                       Shape: (Bs * nimgs, num_tokens, token_dim)
                                       
        Returns:
            torch.Tensor: The predicted scale factor. Shape: (Bs, nimgs, 1)
        """
        Bs, nimgs, C, H, W = depth_feat.shape
        N = enc_tokens.shape[0]
        depth_feat_reshaped = depth_feat.view(N, C, H, W)
        
        # Downsample depth feature using the CNN.
        # Shape: (N, C, H, W) -> (N, 512, 1, 1)
        downsampled_depth = self.depth_cnn_downsampler(depth_feat_reshaped)
        
        # Flatten the CNN output to get a feature vector.
        # Shape: (N, 512, 1, 1) -> (N, 512)
        depth_vec = downsampled_depth.flatten(start_dim=1)
        token_vec = self.token_aggregator(enc_tokens).mean(dim=1)
        
        # Shape: (N, 512 + token_dim) -> (N, 1)
        scale = self.combined_projector(torch.cat([depth_vec, token_vec], dim=1))
        final_scale = scale.view(Bs, nimgs, 1)
        
        return final_scale

