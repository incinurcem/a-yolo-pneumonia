"""
A-YOLO Training Script — Google Colab A100 Optimised + Ablation Support

Yeni ablation flag'leri:
    --mask_strategy {gaussian, random, none}
    --mask_ratio    0.75  (default)
"""

import os
import json
import math
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model   import AYOLO
from dataset2 import AYOLODataset
from loss    import AYOLOLoss
from utils   import AverageMeter, plot_training_curves

# ── A100 / Ampere TF32 speedup ───────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True


# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="A-YOLO Training Script")

    # Paths
    p.add_argument('--img_dir',    type=str, required=True)
    p.add_argument('--train_csv',  type=str, required=True)
    p.add_argument('--val_csv',    type=str, required=True)
    p.add_argument('--output_dir', type=str, default='./outputs')

    # Hyper-parameters
    p.add_argument('--epochs',       type=int,   default=200)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--img_size',     type=int,   default=224)
    p.add_argument('--alpha',        type=float, default=0.2,
                   help="SSL vs Detection loss balance")
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--patience',     type=int,   default=190)

    # 🆕 Ablation flags
    p.add_argument('--mask_strategy', type=str, default='gaussian',
                   choices=['gaussian', 'random', 'none'],
                   help="Ablation: gaussian=anatomy-aware, random=standard MAE, none=no SSL")
    p.add_argument('--mask_ratio',    type=float, default=0.75)

    # Resume
    p.add_argument('--resume',     action='store_true')
    p.add_argument('--checkpoint', type=str, default=None)

    # Hardware
    p.add_argument('--num_workers',     type=int, default=8)
    p.add_argument('--prefetch_factor', type=int, default=2)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def get_cosine_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion,
                    device, epoch, scheduler, scaler):
    model.train()
    meters = {k: AverageMeter() for k in ["total", "recon", "cls", "reg"]}

    pbar = tqdm(loader, desc=f"Epoch {epoch+1:02d} [Train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask_indices"].to(device, non_blocking=True)
        targets = {
            "label": batch["label"].to(device, non_blocking=True).float().view(-1, 1),
            "bbox":  batch["bbox"].to(device,  non_blocking=True).float(),
        }

        with autocast():
            predictions = model(images, masks)
            loss_dict   = criterion(predictions, targets, images, mask_indices=masks)

        scaler.scale(loss_dict["total_loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        bs = images.size(0)
        meters["total"].update(loss_dict["total_loss"].item(), bs)
        meters["recon"].update(loss_dict["recon_loss"].item(), bs)
        meters["cls"].update(loss_dict["cls_loss"].item(),     bs)
        meters["reg"].update(loss_dict["reg_loss"].item(),     bs)

        pbar.set_postfix({
            "loss": f"{meters['total'].avg:.4f}",
            "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    return {k: v.avg for k, v in meters.items()}


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()

    tp, fp, fn, tn = 0, 0, 0, 0

    for batch in tqdm(loader, desc="  [Val]", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask_indices"].to(device, non_blocking=True)
        targets = {
            "label": batch["label"].to(device, non_blocking=True).float().view(-1, 1),
            "bbox":  batch["bbox"].to(device,  non_blocking=True).float(),
        }

        with autocast():
            predictions = model(images, masks)
            loss_dict   = criterion(predictions, targets, images, mask_indices=masks)

        preds  = (torch.sigmoid(predictions["pred_cls"]) > 0.5).float()
        labels = targets["label"]

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()

        loss_meter.update(loss_dict["total_loss"].item(), images.size(0))

    recall    = tp / (tp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-7)

    return {"loss": loss_meter.avg, "acc": accuracy,
            "recall": recall, "precision": precision, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀  Device: {device}")

    # 🆕 Ablation: SSL kapalıysa alpha'yı sıfırla
    if args.mask_strategy == "none":
        args.alpha = 0.0
        print("⚠️  mask_strategy=none → SSL loss kapalı (alpha=0)")
    print(f"🎭  mask_strategy={args.mask_strategy} | ratio={args.mask_ratio} | α={args.alpha}")

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = AYOLODataset(args.train_csv, args.img_dir,
                            img_size=args.img_size, is_train=True,
                            mask_strategy=args.mask_strategy,
                            mask_ratio=args.mask_ratio)
    val_ds   = AYOLODataset(args.val_csv,   args.img_dir,
                            img_size=args.img_size, is_train=False,
                            mask_strategy=args.mask_strategy,
                            mask_ratio=args.mask_ratio)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **loader_kwargs)

    print(f"📦  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model, Optimizer, Loss, Scheduler ────────────────────────────────────
    model     = AYOLO(num_classes=1, img_size=args.img_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    criterion = AYOLOLoss(alpha=args.alpha)
    scaler    = GradScaler()

    num_steps    = args.epochs * len(train_loader)
    warmup_steps = int(0.10 * num_steps)
    scheduler    = get_cosine_warmup_scheduler(optimizer, warmup_steps, num_steps)

    # ── Optional Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    early_stop_counter = 0
    history = []

    if args.resume:
        ckpt_path = args.checkpoint if args.checkpoint else os.path.join(ckpt_dir, "last.pth")
        if os.path.exists(ckpt_path):
            print(f"🔄 Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", [])
            early_stop_counter = ckpt.get("early_stop_counter", 0)

    # 🆕 Ablation config'i kaydet (raporda referans için)
    with open(os.path.join(args.output_dir, "ablation_config.json"), "w") as f:
        json.dump({
            "mask_strategy": args.mask_strategy,
            "mask_ratio":    args.mask_ratio,
            "alpha":         args.alpha,
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "lr":            args.lr,
            "img_size":      args.img_size,
        }, f, indent=2)

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion,
                                        device, epoch, scheduler, scaler)

        val_results = validate(model, val_loader, criterion, device)
        val_loss = val_results["loss"]

        log = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_results.items()},
            "epoch": epoch + 1,
        }
        history.append(log)

        with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        try:
            plot_training_curves(history,
                save_path=os.path.join(args.output_dir, "training_curves.png"))
        except Exception as e:
            print(f"⚠️ Grafik hatası (atlıyorum): {e}")

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Loss: T={train_metrics['total']:.3f} V={val_loss:.3f} | "
            f"Acc: {val_results['acc']*100:.1f}% | "
            f"Recall(Sens): {val_results['recall']:.3f} | "
            f"F1: {val_results['f1']:.3f}"
        )

        # Per-epoch backup (ağır olduğu için isteğe bağlı tutulabilir)
        epoch_ckpt_dir = os.path.join(ckpt_dir, "epoch_backups")
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(epoch_ckpt_dir, f"ayolo_epoch_{epoch+1}.pth"))

        # last.pth
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }, os.path.join(ckpt_dir, "last.pth"))

        # best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, "best_ayolo.pth"))
            print("  ⭐ New Best Model Saved (Loss Improved)!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print("🛑 Early Stopping triggered!")
                break

    # Final save
    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history,
        save_path=os.path.join(args.output_dir, "training_curves.png"))
    print(f"\n✅  Training complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()