"""
A-YOLO Utility Functions
"""

import os
import json
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
class AverageMeter:
    """Tracks running average of a scalar (loss, accuracy, etc.)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history: list[dict], save_path: str):
    """
    Accepts the 'history' list saved by train.py and plots loss / accuracy.

    history entries are expected to contain:
        train_total, train_recon, train_cls, train_reg
        val_loss, val_acc, epoch
    """
    epochs     = [h["epoch"]        for h in history]
    train_loss = [h.get("train_total", h.get("total", 0)) for h in history]
    recon_loss = [h.get("train_recon", h.get("recon", 0)) for h in history]
    val_loss   = [h.get("val_loss",  0) for h in history]
    val_acc    = [h.get("val_acc",   0) for h in history]
    reg_loss   = [h.get("train_reg", h.get("reg", 0)) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss overview
    axes[0].plot(epochs, train_loss, label="Train Total", lw=2)
    axes[0].plot(epochs, val_loss,   label="Val Total",   lw=2, linestyle="--")
    axes[0].plot(epochs, recon_loss, label="SSL Recon",   lw=1.5, alpha=0.8)
    axes[0].set_title("Loss Curves");  axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss");        axes[0].legend();  axes[0].grid(alpha=0.3)

    # 2. Detection regression loss
    axes[1].plot(epochs, reg_loss, color="red", lw=2)
    axes[1].set_title("BBox Regression Loss")
    axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("SmoothL1")
    axes[1].grid(alpha=0.3)

    # 3. Validation accuracy
    axes[2].plot(epochs, [v*100 for v in val_acc], color="teal", lw=2)
    axes[2].set_title("Validation Accuracy")
    axes[2].set_xlabel("Epoch");  axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊  Training curves saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves_from_json(json_path: str, save_dir: str = None):
    """Load history JSON written by train.py and call plot_training_curves."""
    with open(json_path) as f:
        history = json.load(f)
    out_dir  = save_dir or os.path.dirname(json_path)
    out_path = os.path.join(out_dir, "training_curves.png")
    plot_training_curves(history, out_path)


# ─────────────────────────────────────────────────────────────────────────────
def normalize_bbox(bbox, img_shape):
    """[x, y, w, h] → normalised [0, 1].  img_shape = (H, W)"""
    h, w    = img_shape
    x, y, bw, bh = bbox
    return [x/w, y/h, bw/w, bh/h]


def denormalize_bbox(bbox, img_shape):
    """Normalised [0, 1] → pixel [x, y, w, h].  img_shape = (H, W)"""
    h, w    = img_shape
    x, y, bw, bh = bbox
    return [int(x*w), int(y*h), int(bw*w), int(bh*h)]


# ─────────────────────────────────────────────────────────────────────────────
def draw_rsna_boxes(image: np.ndarray,
                    boxes: list,
                    color: tuple = (255, 0, 0)) -> np.ndarray:
    """Draw RSNA-format [x, y, w, h] boxes on a copy of image."""
    out = image.copy()
    for box in boxes:
        x, y, bw, bh = [int(v) for v in box]
        cv2.rectangle(out, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(out, "Opacity", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    torch.save(state, path)
    print(f"  💾  Checkpoint saved: {path}")


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict")))
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0)
