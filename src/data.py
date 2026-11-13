"""Datasets and data loaders for CIFAR-10 (npz) and GTSRB (ppm image folders)."""
import os
import csv
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

DATA_DIR = "data"

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

GTSRB_CLASSES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right-of-way next",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles >3.5t prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware ice/snow", "Wild animals crossing", "End speed+passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End no passing", "End no passing >3.5t",
]
GTSRB_MEAN = (0.485, 0.456, 0.406)
GTSRB_STD = (0.229, 0.224, 0.225)


class CIFAR10NPZ(Dataset):
    def __init__(self, npz_path, train=True, transform=None, indices=None):
        d = np.load(npz_path)
        self.images = d["x_train"] if train else d["x_test"]
        self.labels = d["y_train"] if train else d["y_test"]
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        img = Image.fromarray(self.images[j])
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img, int(self.labels[j])


def _read_gtsrb_gt(path):
    labels = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if not row:
                continue
            labels[row[0]] = int(row[-1])
    return labels


class GTSRBDataset(Dataset):
    def __init__(self, root, split="train", transform=None, val_ratio=0.1, seed=42):
        self.transform = transform
        if split == "test":
            img_dir = os.path.join(root, "Final_Test", "Images")
            gt = _read_gtsrb_gt(os.path.join(root, "Final_Test", "GT-final_test.csv"))
            files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".ppm"))
            self.samples = [(os.path.join(img_dir, f), gt[f]) for f in files]
        else:
            img_dir = os.path.join(root, "Final_Training", "Images")
            classes = sorted(os.listdir(img_dir))
            rng = random.Random(seed)
            self.samples = []
            for c in classes:
                files = sorted(f for f in os.listdir(os.path.join(img_dir, c))
                               if f.lower().endswith(".ppm"))
                items = [(os.path.join(img_dir, c, f), int(c)) for f in files]
                rng.shuffle(items)
                n_val = max(1, int(round(len(items) * val_ratio)))
                if split == "val":
                    self.samples.extend(items[:n_val])
                else:
                    self.samples.extend(items[n_val:])
            self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def cifar_transforms(train):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


def gtsrb_transforms(train, size=64):
    if train:
        return T.Compose([
            T.Resize(size + 8),
            T.RandomCrop(size),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(GTSRB_MEAN, GTSRB_STD),
        ])
    return T.Compose([
        T.Resize(size + 8),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(GTSRB_MEAN, GTSRB_STD),
    ])


def dataset_info(dataset):
    if dataset == "cifar":
        return {"num_classes": 10, "stem": "cifar", "input_size": (3, 32, 32),
                "epochs": 40, "batch_size": 256, "lr": 0.1, "classes": CIFAR_CLASSES}
    if dataset == "gtsrb":
        return {"num_classes": 43, "stem": "imagenet", "input_size": (3, 64, 64),
                "epochs": 15, "batch_size": 128, "lr": 0.05, "classes": GTSRB_CLASSES}
    raise ValueError(dataset)


def build_loaders(args):
    info = dataset_info(args.dataset)
    if args.dataset == "cifar":
        npz = os.path.join(args.data_dir, "cifar-10", "cifar10.npz")
        if not os.path.exists(npz):
            raise FileNotFoundError("CIFAR npz not found: %s (run scripts/download_data.py)" % npz)
        full = CIFAR10NPZ(npz, train=True)
        rng = np.random.RandomState(args.seed)
        train_idx, val_idx = [], []
        for c in range(info["num_classes"]):
            idx = np.where(full.labels == c)[0]
            rng.shuffle(idx)
            n_val = max(1, int(round(len(idx) * args.val_ratio)))
            val_idx.append(idx[:n_val])
            train_idx.append(idx[n_val:])
        train_set = CIFAR10NPZ(npz, train=True, transform=cifar_transforms(True),
                               indices=np.concatenate(train_idx))
        val_set = CIFAR10NPZ(npz, train=True, transform=cifar_transforms(False),
                             indices=np.concatenate(val_idx))
        test_set = CIFAR10NPZ(npz, train=False, transform=cifar_transforms(False))
        info["num_train"], info["num_val"], info["num_test"] = len(train_set), len(val_set), len(test_set)
    elif args.dataset == "gtsrb":
        gtsrb_root = os.path.join(args.data_dir, "GTSRB")
        if not os.path.isdir(gtsrb_root):
            raise FileNotFoundError("GTSRB not found: %s (run scripts/download_data.py)" % gtsrb_root)
        train_set = GTSRBDataset(gtsrb_root, "train", gtsrb_transforms(True), args.val_ratio, args.seed)
        val_set = GTSRBDataset(gtsrb_root, "val", gtsrb_transforms(False), args.val_ratio, args.seed)
        test_set = GTSRBDataset(gtsrb_root, "test", gtsrb_transforms(False), args.val_ratio, args.seed)
        info["num_train"], info["num_val"], info["num_test"] = len(train_set), len(val_set), len(test_set)
    else:
        raise ValueError(args.dataset)

    common = dict(num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0)
    loaders = {
        "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, **common),
        "val": DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **common),
        "test": DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **common),
    }
    return loaders, info
