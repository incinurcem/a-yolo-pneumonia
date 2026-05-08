"""
A-YOLO Inference Script

Usage:
    # Single image
    python inference.py \
        --model_path ./outputs/checkpoints/best_ayolo.pth \
        --img_path   /path/to/image.png \
        --show_gradcam

    # CSV batch
    python inference.py \
        --model_path ./outputs/checkpoints/best_ayolo.pth \
        --img_path   /path/to/test.csv \
        --img_dir    /path/to/stage2_images \
        --show_gradcam
"""

import os
import argparse

import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model          import AYOLO
from explainability import ViTGradCAM, show_cam_on_image

# ImageNet statistics (for denormalisation / display)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="A-YOLO Inference")
    p.add_argument('--model_path',  type=str, required=True)
    p.add_argument('--img_path',    type=str, required=True,
                   help="Image path, directory, or .csv file")
    p.add_argument('--img_dir',     type=str, default=None,
                   help="Image folder (required when img_path is a CSV)")
    p.add_argument('--output_dir',  type=str, default="./inference_results")
    p.add_argument('--img_size',    type=int, default=224)
    p.add_argument('--threshold',   type=float, default=0.5)
    p.add_argument('--show_gradcam', action='store_true')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def load_model(path: str, img_size: int, device: torch.device) -> AYOLO:
    model = AYOLO(num_classes=1, img_size=img_size).to(device)
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
def preprocess(image_path: str, img_size: int, device: torch.device):
    """Load, resize and normalise. Returns (orig_rgb, resized_rgb, tensor)."""
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    orig_rgb    = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    resized     = cv2.resize(orig_rgb, (img_size, img_size))
    normalised  = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor      = torch.from_numpy(normalised).permute(2, 0, 1) \
                       .float().unsqueeze(0).to(device)
    return orig_rgb, resized, tensor


def build_anatomy_mask(img_size: int, patch_size: int, device: torch.device):
    """Build the same Gaussian-weighted anatomy mask used during training."""
    grid = img_size // patch_size
    y, x = np.ogrid[-1:1:complex(grid), -1:1:complex(grid)]
    probs = np.exp(-(x**2 + y**2) / (2 * 0.5**2)).flatten()
    probs = probs / probs.sum()
    idx   = np.random.choice(len(probs), size=int(0.75 * len(probs)),
                              replace=False, p=probs)
    return torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(device)


def denorm(img_norm: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation for display."""
    return np.clip(img_norm * STD + MEAN, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
def process_single_image(model: AYOLO, cam_extractor,
                         image_path: str, args, device: torch.device):
    try:
        orig_rgb, resized, tensor = preprocess(image_path, args.img_size, device)
    except FileNotFoundError as e:
        print(f"⚠️  {e}"); return

    h, w = orig_rgb.shape[:2]
    patch_size = 16
    grid_size  = args.img_size // patch_size

    mask_idx = build_anatomy_mask(args.img_size, patch_size, device)

    # ── Forward ───────────────────────────────────────────────────────────────
    with torch.set_grad_enabled(args.show_gradcam):
        output   = model(tensor, mask_idx)
        pred_cls = torch.sigmoid(output["pred_cls"]).item()
        pred_reg = output["pred_reg"].squeeze().detach().cpu().numpy()

        gradcam_img = None
        if args.show_gradcam and cam_extractor is not None:
            heatmap     = cam_extractor.generate_heatmap(
                output["pred_cls"], img_size=args.img_size)
            gradcam_img = show_cam_on_image(resized.astype(np.float32) / 255,
                                            heatmap)

    # ── Reconstruct masked view for display ──────────────────────────────────
    masked_view = resized.copy()
    for m in mask_idx[0].cpu().numpy():
        r = (m // grid_size) * patch_size
        c = (m %  grid_size) * patch_size
        masked_view[r:r+patch_size, c:c+patch_size] = 0

    # ── Reconstruct SSL output ────────────────────────────────────────────────
    recon = output["reconstruction"]          # [1, 196, P²·3]
    recon_img = recon.view(grid_size, grid_size,
                           patch_size, patch_size, 3) \
                     .permute(0, 2, 1, 3, 4).contiguous() \
                     .view(args.img_size, args.img_size, 3) \
                     .detach().cpu().numpy()
    recon_img = np.clip(denorm(recon_img), 0, 1)

    # ── Plot ──────────────────────────────────────────────────────────────────
    ncols = 4 if args.show_gradcam else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    # Panel 1: Masked Input
    axes[0].imshow(masked_view)
    axes[0].set_title("1. Anatomical Mask\n(Input)", fontweight='bold')
    axes[0].axis("off")

    # Panel 2: SSL Reconstruction
    axes[1].imshow(recon_img)
    axes[1].set_title("2. SSL Reconstruction\n(Model Latent View)",
                       fontweight='bold', color='green')
    axes[1].axis("off")

    col = 2
    # Panel 3: Grad-CAM (optional)
    if args.show_gradcam and gradcam_img is not None:
        axes[col].imshow(gradcam_img)
        axes[col].set_title("3. Grad-CAM\n(Attention Map)",
                             fontweight='bold', color='orange')
        axes[col].axis("off")
        col += 1

    # Panel 4: Final Diagnosis
    axes[col].imshow(orig_rgb)
    label = f"{'PNEUMONIA' if pred_cls > args.threshold else 'NORMAL'}"
    if pred_cls > args.threshold:
        rx, ry, rw, rh = pred_reg
        rect = plt.Rectangle(
            (rx*w, ry*h), rw*w, rh*h,
            linewidth=3, edgecolor='red', facecolor='none')
        axes[col].add_patch(rect)
        axes[col].text(rx*w, max(ry*h - 12, 0),
                       f"{label}: {pred_cls*100:.1f}%",
                       color='white', backgroundcolor='red',
                       fontweight='bold', fontsize=10)
    else:
        axes[col].text(w*0.05, h*0.08, f"{label}: {(1-pred_cls)*100:.1f}%",
                       color='white', backgroundcolor='green',
                       fontweight='bold', fontsize=10)
    axes[col].set_title("4. Final Clinical Diagnosis", fontweight='bold')
    axes[col].axis("off")

    plt.suptitle(os.path.basename(image_path), fontsize=14, y=1.01)
    plt.tight_layout()

    fname = f"result_{os.path.basename(image_path)}"
    save  = os.path.join(args.output_dir, fname)
    plt.savefig(save, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Terminal Logging
    status = "🔴 PNEUMONIA" if pred_cls > args.threshold else "🟢 NORMAL"
    print(f"{status}  ({pred_cls*100:.1f}%)  →  {save}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model         = load_model(args.model_path, args.img_size, device)
    cam_extractor = ViTGradCAM(model) if args.show_gradcam else None

    if args.img_path.endswith('.csv'):
        assert args.img_dir, "--img_dir required when using CSV"
        df = pd.read_csv(args.img_path)

        # 🚀 SADECE TEST SPLIT'İNİ FİLTRELE (Hizalamaya Dikkat!)
        if 'split' in df.columns:
            df = df[df['split'] == 'test'].copy()
            print(f"📂 Only 'test' split selected. Processing {len(df)} images.")
        
        pids = df['patientId'].unique()
        print(f"📊  Processing {len(pids)} patients from CSV...")
        for pid in pids:
            path = os.path.join(args.img_dir, f"{pid}.png")
            process_single_image(model, cam_extractor, path, args, device)

    elif os.path.isdir(args.img_path):
        exts  = ('.png', '.jpg', '.jpeg')
        paths = [os.path.join(args.img_path, f)
                 for f in os.listdir(args.img_path)
                 if f.lower().endswith(exts)]
        print(f"📁  Processing {len(paths)} images from directory...")
        for p in paths:
            process_single_image(model, cam_extractor, p, args, device)

    else:
        process_single_image(model, cam_extractor,
                             args.img_path, args, device)

    if cam_extractor is not None:
        cam_extractor.remove_hooks()


if __name__ == "__main__":
    main()