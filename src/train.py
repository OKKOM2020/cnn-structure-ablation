"""Train / evaluate a CNN on CIFAR-10 or GTSRB.

Examples:
  python -m src.train --dataset cifar --arch resnet18 --tag resnet18_base
  python -m src.train --dataset cifar --arch resnet18 --conv dw --tag resnet18_dw
  python -m src.train --dataset gtsrb --arch resnet18 --pretrained 1 --tag gtsrb_ft

Every run writes a JSON file with the full config, per-epoch history and final
test metrics to results/runs/<tag>.json, plus the best checkpoint.
"""
import argparse
import os
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn

from .data import build_loaders, dataset_info
from .models import build_model
from .engine import fit, evaluate, build_optimizer, make_scheduler
from .utils import (set_seed, count_parameters, count_macs, save_json, load_json)

# keep torchvision pretrained weights inside the project (sandbox may forbid ~/.cache)
os.environ.setdefault("TORCH_HOME", os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "torch_cache")))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cifar", "gtsrb"])
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "lenet", "vgg11"])
    p.add_argument("--conv", default="standard", choices=["standard", "dw"])
    p.add_argument("--attn", default="none", choices=["none", "se", "cbam"])
    p.add_argument("--act", default="relu", choices=["relu", "gelu", "mish"])
    p.add_argument("--norm", default="bn", choices=["bn", "gn"])
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--pretrained", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results/runs")
    p.add_argument("--ckpt-dir", default="results/checkpoints")
    p.add_argument("--tag", default="")
    p.add_argument("--limit-train-batches", type=int, default=0)
    p.add_argument("--limit-val-batches", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def load_imagenet_weights(model, arch="resnet18"):
    import torchvision.models as tvm
    if arch == "resnet18":
        tv = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    elif arch == "resnet34":
        tv = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("pretrained only supported for resnet18/resnet34")
    state = tv.state_dict()
    state.pop("fc.weight", None)
    state.pop("fc.bias", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[pretrained] loaded ImageNet weights; missing=%d unexpected=%d"
          % (len(missing), len(unexpected)))
    return missing, unexpected


def main(argv=None):
    args = parse_args(argv)
    info = dataset_info(args.dataset)
    if args.epochs <= 0:
        args.epochs = info["epochs"]
    if args.batch_size <= 0:
        args.batch_size = info["batch_size"]
    if args.lr <= 0:
        args.lr = info["lr"]
    if args.smoke:
        args.epochs = 1
        args.limit_train_batches = args.limit_train_batches or 8
        args.limit_val_batches = args.limit_val_batches or 4
        args.workers = 0
        if not args.tag.startswith("smoke"):
            args.tag = "smoke_" + (args.tag or "run")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    loaders, info = build_loaders(args)
    model = build_model(args.arch, num_classes=info["num_classes"], stem=info["stem"],
                        conv=args.conv, attn=args.attn, act=args.act, norm=args.norm)
    if args.pretrained:
        load_imagenet_weights(model, args.arch)
    model = model.to(device)

    params = count_parameters(model)
    macs = count_macs(model, info["input_size"])
    print("=" * 70)
    print("dataset=%s arch=%s conv=%s attn=%s act=%s norm=%s stem=%s"
          % (args.dataset, args.arch, args.conv, args.attn, args.act, args.norm, info["stem"]))
    print("train/val/test = %d/%d/%d | params=%.3fM | MACs=%.3fG"
          % (info["num_train"], info["num_val"], info["num_test"],
             params / 1e6, macs / 1e9))
    print("device=%s | epochs=%d batch=%d lr=%g wd=%g amp=%d"
          % (device, args.epochs, args.batch_size, args.lr, args.wd, args.amp))
    print("=" * 70, flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    args.scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    t0 = time.time()
    history, best_val, best_epoch = fit(model, loaders, criterion, optimizer,
                                        scheduler, device, args)
    test = evaluate(model, loaders["test"], nn.CrossEntropyLoss(), device, collect=True)
    elapsed = time.time() - t0

    # extra metrics
    preds = np.array(test.pop("preds"))
    labels = np.array(test.pop("labels"))
    num_classes = info["num_classes"]
    acc = float((preds == labels).mean() * 100.0)
    f1s = []
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        conf[int(t), int(p)] += 1
    for c in range(num_classes):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(float(f1))
    macro_f1 = float(np.mean(f1s))

    tag = args.tag or ("%s_%s_%s_%s_%s_%s" % (args.dataset, args.arch, args.conv,
                                              args.attn, args.act, args.norm))
    result = {
        "tag": tag,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "dataset": args.dataset, "arch": args.arch, "stem": info["stem"],
            "conv": args.conv, "attn": args.attn, "act": args.act, "norm": args.norm,
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "wd": args.wd, "label_smoothing": args.label_smoothing,
            "warmup_epochs": args.warmup_epochs, "seed": args.seed,
            "val_ratio": args.val_ratio, "pretrained": bool(args.pretrained),
            "amp": bool(args.amp),
        },
        "data": {"num_train": info["num_train"], "num_val": info["num_val"],
                 "num_test": info["num_test"], "num_classes": info["num_classes"]},
        "model": {"params": int(params), "params_m": params / 1e6,
                  "macs": int(macs), "macs_g": macs / 1e9},
        "best_val_acc1": float(best_val),
        "best_epoch": int(best_epoch),
        "test": {"acc1": test["acc1"], "acc5": test["acc5"],
                 "loss": test["loss"], "macro_f1": macro_f1,
                 "per_class_f1": f1s, "confusion": conf.tolist()},
        "history": history,
        "elapsed_sec": round(elapsed, 1),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    out_path = os.path.join(args.out_dir, tag + ".json")
    save_json(result, out_path)
    ckpt_path = os.path.join(args.ckpt_dir, tag + ".pt")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": result["config"]}, ckpt_path)
    print("-" * 70)
    print("BEST val acc1 = %.2f%% (epoch %d)" % (best_val, best_epoch))
    print("TEST acc1 = %.2f%%  acc5 = %.2f%%  macroF1 = %.4f"
          % (test["acc1"], test["acc5"], macro_f1))
    print("params = %.3fM  MACs = %.3fG  elapsed = %.1fs"
          % (params / 1e6, macs / 1e9, elapsed))
    print("saved:", out_path)
    print("saved:", ckpt_path, flush=True)
    return result


if __name__ == "__main__":
    main()
