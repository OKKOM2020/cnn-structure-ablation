"""VGG11-BN adapted to 32x32 inputs with global-average-pool head."""
import torch.nn as nn
from .layers import DepthwiseSeparableConv, get_act, get_norm

VGG11_CFG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG11(nn.Module):
    def __init__(self, num_classes=10, conv="standard", act="relu", norm="bn", attn="none"):
        super().__init__()
        layers = []
        in_ch = 3
        for v in VGG11_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                continue
            if conv == "standard":
                layers.append(nn.Conv2d(in_ch, v, 3, 1, 1, bias=False))
            elif conv == "dw":
                layers.append(DepthwiseSeparableConv(in_ch, v, 3, 1, 1))
            else:
                raise ValueError(conv)
            layers.append(get_norm(norm, v))
            layers.append(get_act(act))
            in_ch = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)
