"""
based on https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
"""
import math

import torch
import torch.nn as nn


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att



class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, decision_mask):
        B, C, H, W = x.shape

        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        spatial_feat = torch.cat([avg_out, max_out], dim=1)  # (B,2,H,W)

        spatial_att_logit = self.conv(spatial_feat)  # (B,1,H,W)

        spatial_att = self.sigmoid(spatial_att_logit)  # (B,1,H,W)

        if decision_mask is not None:
            if decision_mask.dim() == 2:  # (B, N) -> (B,1,H,W)
                decision_mask = decision_mask.view(B, 1, H, W)
            decision_mask = decision_mask.to(spatial_att.dtype).to(spatial_att.device)
            spatial_att = spatial_att * decision_mask

        return x * spatial_att, spatial_att

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio)
        self.spatial_gate = SpatialGate(kernel_size)

    def forward(self, x, decision_mask=None):
        B = x.shape[0]
        if x.dim() == 3:  # (B, N, C) → (B, C, H, W)
            N, C = x.shape[1], x.shape[2]
            H = W = int(math.sqrt(N))
            x = x.permute(0, 2, 1).view(B, C, H, W)

        x = self.channel_gate(x)
        x, spatial_att = self.spatial_gate(x, decision_mask)

        N = H * W
        x = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        spatial_att = spatial_att.view(B, N)
        return x, spatial_att