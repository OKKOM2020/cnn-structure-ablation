"""LeNet-5 adapted to 3x32x32 inputs (classic shallow baseline)."""
import torch.nn as nn
from .layers import get_act, get_norm


class LeNet(nn.Module):
    def __init__(self, num_classes=10, act="relu", norm="bn"):
        super().__init__()
        act1 = get_act(act)
        act2 = get_act(act)
        act3 = get_act(act)
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5, bias=False),
            get_norm(norm, 6), act1, nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, bias=False),
            get_norm(norm, 16), act2, nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), act3,
            nn.Linear(120, 84), get_act(act),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
