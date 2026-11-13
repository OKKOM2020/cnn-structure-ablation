"""ResNet for CIFAR (3x3 stem) and ImageNet-style (7x7 stem) inputs.

Knobs for the ablation study:
  conv : standard 3x3 conv | depthwise-separable conv
  attn : none | se | cbam  (applied after the second conv of every block)
  act  : relu | gelu | mish
  norm : bn | gn
"""
import torch.nn as nn
from .layers import DepthwiseSeparableConv, get_act, get_norm, get_attention


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv="standard", attn="none", act="relu", norm="bn"):
        super().__init__()
        if conv == "standard":
            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        elif conv == "dw":
            self.conv1 = DepthwiseSeparableConv(in_planes, planes, 3, stride, 1)
            self.conv2 = DepthwiseSeparableConv(planes, planes, 3, 1, 1)
        else:
            raise ValueError("unknown conv type: " + str(conv))
        self.bn1 = get_norm(norm, planes)
        self.bn2 = get_norm(norm, planes)
        self.act1 = get_act(act)
        self.act2 = get_act(act)
        self.attn = get_attention(attn, planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                get_norm(norm, planes),
            )

    def forward(self, x):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.attn(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.act2(out + identity)
        return out


class ResNet(nn.Module):
    def __init__(self, layers, num_classes=10, stem="cifar", conv="standard",
                 attn="none", act="relu", norm="bn"):
        super().__init__()
        self._conv = conv
        self._attn = attn
        self._act = act
        self._norm = norm

        if stem == "cifar":
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.maxpool = nn.Identity()
        elif stem == "imagenet":
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
        else:
            raise ValueError("unknown stem: " + str(stem))
        self.bn1 = get_norm(norm, 64)
        self.relu = get_act(act)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, layers[0], 1)
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self._init_params()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(BasicBlock(self.in_planes, planes, s,
                                     self._conv, self._attn, self._act, self._norm))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)
