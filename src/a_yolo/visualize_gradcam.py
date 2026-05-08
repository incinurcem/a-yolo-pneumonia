"""
A-YOLO Visual Explainer – 4-Panel Grid (CSV batch)

Usage:
    python visualize_gradcam.py \
        --model_path ./outputs/checkpoints/best_ayolo.pth \
        --csv_path   /path/to/test.csv \
        --img_dir    /path/to/stage2_images \
        --num_samples 20
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

# ImageNet statistics for denormalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="A-YOLO Visual Explanation Generator")
    p.add_argument('--model_path',  type=str, required=True)
    p.add_argument('--csv_path',    type=str, required=True)
    p.add_argument('--img_dir',     type=str, required=True)
    p.add_argument('--output_dir',  type=str, default="./visual_explanations")
    p.add_argument('--img_size',    type=int, default=224)
    p.add_argument('--num_samples', type=int, default=10,
                   help="Number of random samples to visualize")
    p.add_argument('--balanced',    action='store_true',
                   help="Sample equal numbers of positive/negative cases")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def visualise_one(model: AYOLO, cam: ViTGradCAM,
                  img_id: str, img_dir: str,
                  img_size: int, output_dir: str,
                  device: torch.device):
    # 1. Load and Preprocess
    img_path = os.path.join(img_dir, f"{img_id}.png")
    orig     = cv2.imread(img_path)
    if orig is None:
        print(f"⚠️  Image not found: {img_path}");  return

    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w     = orig_rgb.shape[:2]
    patch_sz = 16
    grid_sz  = img_size // patch_sz

    resized  = cv2.resize(orig_rgb, (img_size, img_size))
    norm     = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor   = torch.from_numpy(norm).permute(2, 0, 1) \
                    .float().unsqueeze(0).to(device)

    # 2. Build Anatomy Mask (Gaussian-weighted)
    y, x = np.ogrid[-1:1:complex(grid_sz), -1:1:complex(grid_sz)]
    probs = np.exp(-(x**2 + y**2) / (2 * 0.6**2)).flatten()
    probs = probs / probs.sum()
    idx   = np.random.choice(len(probs), int(0.75*len(probs)),
                              replace=False, p=probs)
    mask_t = torch.tensor(idx, dtype=torch.long).unsqueeze(0).to(device)

    # 3. Forward Pass + Grad-CAM Generation
    with torch.set_grad_enabled(True):
        out      = model(tensor, mask_t)
        heatmap  = cam.generate_heatmap(out["pred_cls"], img_size=img_size)
        grad_img = show_cam_on_image(resized.astype(np.float32)/255, heatmap)
        pred_cls = torch.sigmoid(out["pred_cls"]).item()
        pred_reg = out["pred_reg"].squeeze().detach().cpu().numpy()

    # 4. Generate Masked View for visualization
    masked = resized.copy()
    for m in idx:
        r = (m // grid_sz) * patch_sz
        c = (m %  grid_sz) * patch_sz
        masked[r:r+patch_sz, c:c+patch_sz] = 0

    # 5. SSL Reconstruction (Latent View)
    recon_raw = out["reconstruction"] \
        .view(grid_sz, grid_sz, patch_sz, patch_sz, 3) \
        .permute(0, 2, 1, 3, 4).contiguous() \
        .view(img_size, img_size, 3).detach().cpu().numpy()
    recon = np.clip(recon_raw * STD + MEAN, 0, 1)

    # 6. Create 4-Panel Visualization Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: Masked Input
    axes[0].imshow(masked)
    axes[0].set_title("1. Anatomical Mask\n(Input)", fontweight='bold')
    axes[0].axis("off")

    # Panel 2: SSL Reconstruction
    axes[1].imshow(recon)
    axes[1].set_title("2. SSL Reconstruction\n(Latent Recovery)",
                       fontweight='bold', color='green')
    axes[1].axis("off")

    # Panel 3: Grad-CAM
    axes[2].imshow(grad_img)
    axes[2].set_title("3. Grad-CAM\n(Attention Map)",
                       fontweight='bold', color='darkorange')
    axes[2].axis("off")

    # Panel 4: Final Prediction
    axes[3].imshow(orig_rgb)
    is_positive = pred_cls > 0.5
    colour = "red" if is_positive else "green"
    label  = "PNEUMONIA" if is_positive else "NORMAL"
    
    if is_positive:
        rx, ry, rw, rh = pred_reg
        rect = plt.Rectangle((rx*w, ry*h), rw*w, rh*h,
                              lw=3, edgecolor='red', facecolor='none')
        axes[3].add_patch(rect)
        
    axes[3].set_title(f"4. Diagnosis: {label}\nConfidence: {pred_cls*100:.1f}%",
                       fontweight='bold', color=colour)
    axes[3].axis("off")

    plt.suptitle(f"A-YOLO Explanation Report: {img_id}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the result
    out_path = os.path.join(output_dir, f"explain_{img_id}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ Generated: {img_id} → {label} ({pred_cls*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize and load model
    model = AYOLO(num_classes=1, img_size=args.img_size).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    cam = ViTGradCAM(model)

    # Handle Patient Sampling from CSV
    df = pd.read_csv(args.csv_path)
    # Detect target column (handle case sensitivity)
    target_col = 'target' if 'target' in df.columns else 'Target'
    
    if args.balanced:
        pos = df[df[target_col] == 1]['patientId'].unique()
        neg = df[df[target_col] == 0]['patientId'].unique()
        half = args.num_samples // 2
        sample_ids = np.concatenate([
            np.random.choice(pos, min(half, len(pos)), replace=False),
            np.random.choice(neg, min(half, len(neg)), replace=False),
        ])
    else:
        all_ids    = df['patientId'].unique()
        sample_ids = np.random.choice(
            all_ids, min(args.num_samples, len(all_ids)), replace=False)

    print(f"🖼️  Generating explanations for {len(sample_ids)} samples...")
    for pid in sample_ids:
        visualise_one(model, cam, pid, args.img_dir,
                      args.img_size, args.output_dir, device)

    cam.remove_hooks()
    print(f"\n🎉 Process Complete. Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()