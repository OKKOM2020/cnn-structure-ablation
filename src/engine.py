"""Training / evaluation loops with AMP and cosine LR schedule."""
import time
import math
import torch
from .utils import AverageMeter, accuracy


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, max_batches=0):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    n_batches = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            output = model(images)
            loss = criterion(output, targets)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        acc1 = accuracy(output, targets)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    return losses.avg, top1.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=0, collect=False):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    preds, labels = [], []
    n_batches = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        output = model(images)
        loss = criterion(output, targets)
        a1, a5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(a1, images.size(0))
        top5.update(a5, images.size(0))
        if collect:
            preds.append(output.argmax(1).cpu())
            labels.append(targets.cpu())
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    out = {"loss": losses.avg, "acc1": top1.avg, "acc5": top5.avg}
    if collect and preds:
        out["preds"] = torch.cat(preds).numpy().tolist()
        out["labels"] = torch.cat(labels).numpy().tolist()
    return out


def make_scheduler(optimizer, epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def fit(model, loaders, criterion, optimizer, scheduler, device, args, logger=print):
    history = []
    best_acc = -1.0
    best_epoch = 0
    best_state = None
    max_train = args.limit_train_batches
    max_eval = args.limit_val_batches
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders["train"], criterion, optimizer,
                                          device, args.scaler, max_train)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        if max_eval and max_eval < len(loaders["val"]):
            val = evaluate(model, loaders["val"], criterion, device, max_eval)
        else:
            val = evaluate(model, loaders["val"], criterion, device)
        improved = val["acc1"] > best_acc
        if improved:
            best_acc = val["acc1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "lr": round(lr_now, 6),
               "train_loss": round(tr_loss, 4), "train_acc1": round(tr_acc, 3),
               "val_loss": round(val["loss"], 4), "val_acc1": round(val["acc1"], 3),
               "val_acc5": round(val["acc5"], 3), "epoch_time": round(time.time() - t0, 1)}
        history.append(row)
        logger("epoch %3d/%d | lr %.4f | train loss %.4f acc %.2f | val loss %.4f acc %.2f | %.1fs%s"
               % (epoch, args.epochs, lr_now, tr_loss, tr_acc, val["loss"], val["acc1"],
                  time.time() - t0, "  *" if improved else ""))
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_acc, best_epoch


def build_optimizer(model, args):
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                           weight_decay=args.wd, nesterov=True)
