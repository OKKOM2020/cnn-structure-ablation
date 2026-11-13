# 基于经典 CNN 的结构改造与交通标志识别

## 项目简介

本仓库实现了一个可配置的 ResNet-18：卷积分支可在标准卷积与深度可分离卷积之间切换，可插入 SE/CBAM 注意力，激活函数可选 ReLU/GELU/Mish，归一化层可选 BatchNorm/GroupNorm。围绕该网络完成两组实验：

1. CIFAR-10 结构消融：分别替换卷积、注意力、激活函数与归一化层，在统一训练协议下对比参数量、计算量（MACs）与测试精度；
2. GTSRB 交通标志场景验证：对比从头训练、ImageNet 预训练微调与轻量版（DW+SE+GELU）迁移，并分析 CPU/GPU 单张延迟、模型体积、十种图像退化下的鲁棒性与不同输入分辨率的影响。

主要结果（CIFAR-10 测试集，40 轮统一协议）：ResNet-18 基线 93.19%；深度可分离卷积使参数量与计算量下降约 87%，精度 92.09%；SE 93.19%、CBAM 92.49%、GELU 93.07%、Mish 92.62%、GroupNorm 91.33%、DW+SE+GELU 91.47%；对照模型 VGG11-BN 89.34%、LeNet 74.08%。GTSRB 上从头训练 98.50%、预训练微调 98.61%、轻量版 97.65%（参数约为基线的 1/7）。

## 代码结构

    src/
    ├── data.py      CIFAR-10(npz) / GTSRB(ppm) 数据管道与数据增强
    ├── engine.py    训练/评估循环、AMP、余弦退火与 warmup
    ├── train.py     统一训练入口（数据集、结构开关、超参数、结果保存）
    ├── utils.py     参数量与 MACs 统计、指标、JSON 工具
    └── models/
        ├── layers.py   深度可分离卷积、SE、CBAM、激活与归一化组件
        ├── resnet.py   ResNet-18/34（CIFAR 与 ImageNet 两种 stem）
        ├── lenet.py
        └── vgg.py

## 数据与权重

仓库不包含数据与权重文件，下载后放入对应目录（data/ 与 results/ 已加入 .gitignore）：

| 文件 | SHA256 | 放置方式 |
|---|---|---|
| GTSRB.tar.gz | 76743944c28d7429fd1d14ccbb961647dfc29289cdda81ec0c0e2e2be6c879c9 | 解压到 data/ |
| cifar10.npz | 27d3cc7c2c65fe18d7d11df0f96e5914faa5294ad5cda9c3fd882d6c03e4f3da | 放到 data/cifar-10/cifar10.npz |
| checkpoints.zip | 7c31f9117985b21afe6a4711a6ccb7e5deb00ab487c2c919f8946a7b3072b369 | 解压到 results/（得到 results/checkpoints/） |

下载地址：

- https://github.com/OKKOM2020/cnn-structure-ablation/releases/latest/download/GTSRB.tar.gz
- https://github.com/OKKOM2020/cnn-structure-ablation/releases/latest/download/cifar10.npz
- https://github.com/OKKOM2020/cnn-structure-ablation/releases/latest/download/checkpoints.zip

数据格式：

- data/cifar-10/cifar10.npz：包含 x_train (50000, 32, 32, 3)、y_train、x_test (10000, 32, 32, 3)、y_test。
- data/GTSRB/：Final_Training/Images/00000 … 00042（39209 张），Final_Test/Images 与 GT-final_test.csv（12630 张）。
- checkpoints.zip 为 13 组实验的最佳权重，每个 .pt 含 model（state_dict）与 config（实验配置）两个键。

## 环境

    Python 3.12 / PyTorch 2.6.0+cu124 / torchvision 0.21.0
    pip install -r requirements.txt

## 训练

    python -m src.train --dataset cifar --arch resnet18 --tag my_run
    python -m src.train --dataset cifar --arch resnet18 --conv dw --attn se --act gelu --tag my_light
    python -m src.train --dataset gtsrb --arch resnet18 --epochs 15 --lr 0.05 --tag gtsrb_scratch
    python -m src.train --dataset gtsrb --arch resnet18 --pretrained 1 --epochs 10 --lr 0.01 --tag gtsrb_ft

训练结果（逐轮曲线、测试集指标、混淆矩阵）写入 results/runs/。可选参数：--conv dw、--attn se|cbam、--act gelu|mish、--norm bn|gn、--pretrained 1 等，完整列表见 python -m src.train --help。

## 说明

训练默认开启混合精度（AMP）；无 GPU 时自动使用 CPU；Windows 受限环境下 DataLoader 多进程可能报 WinError 5，可加 --workers 0。
