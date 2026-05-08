"""
A-YOLO Evaluation Script

Outputs:
  • Accuracy, Precision, Recall, F1, AUC-ROC, mAP
  • IoU distribution (positive samples)
  • Confusion matrix, ROC curve, PR curve
  • detailed_eval_results.csv

Usage:
    python evaluate.py \
        --model_path ./outputs/checkpoints/best_ayolo.pth \
        --img_dir    /path/to/stage2_images \
        --csv_path   /path/to/test.csv \
        --output_dir ./eval_results
"""

import os
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
)

from model   import AYOLO
from dataset import AYOLODataset


# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--img_dir',    type=str, required=True)
    p.add_argument('--csv_path',   type=str, required=True)
    p.add_argument('--output_dir', type=str, default='./eval_results')
    p.add_argument('--img_size',   type=int, default=224)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--threshold',  type=float, default=0.5)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def calculate_iou(box_pred: np.ndarray, box_true: np.ndarray, img_size: int = 224) -> float:
    """
    Kökten çözüm: Her şeyi 0-1 aralığında bırak, piksellere hiç girme!
    """
    # Her iki kutunun da [0, 1] aralığında olduğundan emin olalım
    # Dataloader zaten normalize ediyor (box_true). 
    # Model de sigmoid/tanh/linear sonrası normalize çıkıyor (box_pred).

    # Pred: [x, y, w, h]
    p_x1, p_y1 = box_pred[0], box_pred[1]
    p_x2, p_y2 = box_pred[0] + box_pred[2], box_pred[1] + box_pred[3]

    # True: [x, y, w, h]
    t_x1, t_y1 = box_true[0], box_true[1]
    t_x2, t_y2 = box_true[0] + box_true[2], box_true[1] + box_true[3]

    # Kesişim (Intersection)
    ix1 = max(p_x1, t_x1);  iy1 = max(p_y1, t_y1)
    ix2 = min(p_x2, t_x2);  iy2 = min(p_y2, t_y2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    
    # Birleşim (Union)
    area_p = box_pred[2] * box_pred[3]
    area_t = box_true[2] * box_true[3]
    union_area = area_p + area_t - inter_area

    # Güvenlik: Sıfıra bölünme hatası
    if union_area <= 0:
        return 0.0
        
    return float(inter_area / union_area)

# ─────────────────────────────────────────────────────────────────────────────
def plot_curves(y_true, y_prob, output_dir: str):
    fpr, tpr, _     = roc_curve(y_true, y_prob)
    roc_auc          = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score         = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    axes[0].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0,1], [0,1], 'navy', lw=1.5, linestyle='--')
    axes[0].set_xlabel('FPR');  axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve');  axes[0].legend()

    # PR
    axes[1].plot(recall, precision, color='blue', lw=2,
                 label=f'AP = {ap_score:.3f}')
    axes[1].set_xlabel('Recall');  axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve');  axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_curves.png"), dpi=150)
    plt.close()
    return roc_auc, ap_score


# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = AYOLO(num_classes=1, img_size=args.img_size).to(device)
    ckpt  = torch.load(args.model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"✅  Model loaded from {args.model_path}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_ds = AYOLODataset(args.csv_path, args.img_dir, 
                       img_size=args.img_size, is_train=False, 
                       split_type="test") 
    loader  = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=2)
    print(f"📊  Evaluating {len(test_ds)} samples...")

    # Initialize lists (Required for calculations)
    rows        = []
    all_probs   = []
    all_true    = []
    all_ious    = []
    y_prob_det  = [] # For mAP calculation
    y_true_det  = [] # For mAP calculation

    with torch.no_grad():
        for batch in tqdm(loader):
            imgs    = batch["image"].to(device)
            outputs = model(imgs, mask_indices=None)

            probs       = torch.sigmoid(outputs["pred_cls"]).cpu().numpy().flatten()
            true_labels = batch["label"].numpy().flatten()
            true_boxes  = batch["bbox"].numpy()
            pred_boxes  = outputs["pred_reg"].cpu().numpy()

            positives_in_batch = (batch["label"] == 1).sum().item()
          

            for i in range(len(true_labels)):
                iou = 0.0
                if true_labels[i] == 1:
             
                    iou = calculate_iou(pred_boxes[i], true_boxes[i], img_size=args.img_size)
                  
                    all_ious.append(iou)

                all_probs.append(float(probs[i]))
                all_true.append(int(true_labels[i]))

                # 🚀 mAP (AP@50) Logic:
                # A detection is a "True Positive" if the class is correct (1) 
                # AND the bounding box IoU >= 0.5.
                is_correct_detection = 1 if (true_labels[i] == 1 and iou >= 0.5) else 0
                
                y_prob_det.append(float(probs[i]))
                y_true_det.append(is_correct_detection)

                rows.append({
                    "patientId": batch["patientId"][i],
                    "target":    int(true_labels[i]),
                    "pred":      int(probs[i] > args.threshold),
                    "prob":      float(probs[i]),
                    "iou":       float(iou),
                })

    # ── Metrics ───────────────────────────────────────────────────────────────
    df     = pd.DataFrame(rows)
    y_true = df["target"].values.astype(int)
    y_pred = (df["prob"].values > args.threshold).astype(int) 
    y_prob = df["prob"].values.astype(float)

    # mAP (Average Precision @ IoU 0.5)
    map50 = average_precision_score(y_true_det, y_prob_det)

    # Standard Metrics
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    avg_method = "binary" if len(unique_labels) <= 2 else "macro"
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg_method, zero_division=0)
    
    acc      = (y_true == y_pred).mean()
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    iou50    = (np.array(all_ious) >= 0.5).mean() if all_ious else 0.0

    # Plotting
    roc_auc, ap_score = plot_curves(y_true, y_prob, args.output_dir)

    print("\n" + "═"*50)
    print("        🚀 A-YOLO EVALUATION REPORT")
    print("═"*50)
    print(f"  ✨ Accuracy     : {acc*100:.2f}%")
    print(f"  🎯 Precision    : {prec:.4f}")
    print(f"  🔎 Recall       : {rec:.4f}  (Sensitivity)")
    print(f"  ⚖️  F1-Score     : {f1:.4f}")
    print(f"  📈 AUC-ROC      : {roc_auc:.4f}")
    print(f"  📊 mAP (AP@50)  : {map50:.4f}  🚀")
    print(f"  📏 Mean IoU     : {mean_iou:.4f}")
    print(f"  ✅ IoU >= 0.5   : {iou50*100:.1f}%")
    print("═"*50 + "\n")

    
    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Pneumonia"],
                yticklabels=["Normal", "Pneumonia"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Ground Truth", fontsize=12)
    plt.title("Confusion Matrix — Pneumonia Detection Performance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── IoU Distribution ─────────────────────────────────────────────────────
    if all_ious:
        plt.figure(figsize=(9, 5))
        sns.histplot(all_ious, bins=25, kde=True, color="teal", alpha=0.6)
        plt.axvline(0.5, color="crimson", linestyle="--", linewidth=2, label="IoU=0.5 Threshold")
        plt.title(f"Bbox IoU Distribution (Mean={mean_iou:.3f})", fontsize=14)
        plt.xlabel("IoU (Intersection over Union)");  plt.ylabel("Sample Count")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "iou_distribution.png"), dpi=150)
        plt.close()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df.to_csv(os.path.join(args.output_dir, "detailed_eval_results.csv"), index=False)
    print(f"🎉 All results and plots saved to: {args.output_dir}")


if __name__ == "__main__":
    run_evaluation()