from .resnet import ResNet, BasicBlock
from .lenet import LeNet
from .vgg import VGG11


def build_model(arch, num_classes=10, stem="cifar", conv="standard", attn="none", act="relu", norm="bn"):
    if arch == "resnet18":
        return ResNet([2, 2, 2, 2], num_classes=num_classes, stem=stem,
                      conv=conv, attn=attn, act=act, norm=norm)
    if arch == "resnet34":
        return ResNet([3, 4, 6, 3], num_classes=num_classes, stem=stem,
                      conv=conv, attn=attn, act=act, norm=norm)
    if arch == "lenet":
        return LeNet(num_classes=num_classes, act=act, norm=norm)
    if arch == "vgg11":
        return VGG11(num_classes=num_classes, conv=conv, act=act, norm=norm, attn=attn)
    raise ValueError("unknown arch: " + str(arch))
