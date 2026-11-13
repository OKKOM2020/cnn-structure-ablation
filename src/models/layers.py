"""Reusable CNN building blocks: activations, normalizations, separable conv, attention."""
import torch
import torch.nn as nn


def get_act(name):
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    raise ValueError("unknown activation: " + str(name))


def get_norm(name, channels):
    name = (name or "bn").lower()
    if name == "bn":
        return nn.BatchNorm2d(channels)
    if name == "gn":
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    if name == "none":
        return nn.Identity()
    raise ValueError("unknown norm: " + str(name))


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise convolution followed by 1x1 pointwise convolution (no bias)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                            groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True)[0]
        y = self.conv(torch.cat([avg_map, max_map], dim=1))
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


def get_attention(name, channels):
    name = (name or "none").lower()
    if name == "none":
        return nn.Identity()
    if name == "se":
        return SELayer(channels)
    if name == "cbam":
        return CBAM(channels)
    raise ValueError("unknown attention: " + str(name))
