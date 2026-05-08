"""
A-YOLO Advanced Analysis: ECE Calibration, Error Taxonomy, MC Dropout Uncertainty.

Senin gerçek model çıktına göre uyarlandı:
    outputs["pred_cls"], outputs["pred_reg"], batch["bbox"], batch["label"]
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from model   import AYOLO
from dataset2 import AYOLODataset


# ─────────────────────────────────────────────────────────────────────────────
def enable_dropout(model):
    """MC Dropout için tüm Dropout katmanlarını train moduna alır."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


# ─────────────────────────────────────────────────────────────────────────────
def calculate_iou(box_pred, box_true):
    """COCO formatında [x, y, w, h] kutularda IoU."""
    p_x1, p_y1 = box_pred[0], box_pred[1]
    p_x2, p_y2 = box_pred[0] + box_pred[2], box_pred[1] + box_pred[3]
    t_x1, t_y1 = box_true[0], box_true[1]
    t_x2, t_y2 = box_true[0] + box_true[2], box_true[1] + box_true[3]

    iw = max(0.0, min(p_x2, t_x2) - max(p_x1, t_x1))
    ih = max(0.0, min(p_y2, t_y2) - max(p_y1, t_y1))
    inter = iw * ih
    union = box_pred[2] * box_pred[3] + box_true[2] * box_true[3] - inter
    return float(inter / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
def calculate_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error — manuel binning (sklearn'e bağlı değil)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_acc, bin_conf = [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(acc - conf)
        bin_acc.append(acc)
        bin_conf.append(conf)
    return float(ece), np.array(bin_acc), np.array(bin_conf)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--csv_path",   type=str, required=True)
    p.add_argument("--img_dir",    type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mc_passes",  type=int, default=20)
    p.add_argument("--threshold",  type=float, default=0.45)
    p.add_argument("--iou_thresh", type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ──────────────────────────────────────────────────────────────
    model = AYOLO(num_classes=1, img_size=args.img_size).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    print(f"✅ Model: {args.model_path}")

    # ── Dataset ────────────────────────────────────────────────────────────
    ds = AYOLODataset(args.csv_path, args.img_dir,
                      img_size=args.img_size, is_train=False, split_type="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"📊 {len(ds)} örnek üzerinde gelişmiş analiz başlıyor...")

    y_true_all, y_prob_all, uncertainties = [], [], []
    err = {
        "True_Positive":      0,
        "Localization_Error": 0,
        "Background_Error":   0,
        "False_Negative":     0,
        "True_Negative":      0,
    }

    for batch in tqdm(loader, desc="Analyzing"):
        imgs   = batch["image"].to(device)
        labels = batch["label"].numpy().flatten()
        tboxes = batch["bbox"].numpy()

        # 1) Standart inference
        with torch.no_grad():
            out = model(imgs, mask_indices=None)
            probs  = torch.sigmoid(out["pred_cls"]).cpu().numpy().flatten()
            pboxes = out["pred_reg"].cpu().numpy()

        for i in range(len(labels)):
            y_prob_all.append(float(probs[i]))
            y_true_all.append(int(labels[i]))

            pred_pos = probs[i] > args.threshold
            has      = labels[i] == 1

            if pred_pos and has:
                iou = calculate_iou(pboxes[i], tboxes[i])
                if iou >= args.iou_thresh:
                    err["True_Positive"] += 1
                else:
                    err["Localization_Error"] += 1
            elif pred_pos and not has:
                err["Background_Error"] += 1
            elif not pred_pos and has:
                err["False_Negative"] += 1
            else:
                err["True_Negative"] += 1

        # 2) MC Dropout
        enable_dropout(model)
        mc_probs = []
        with torch.no_grad():
            for _ in range(args.mc_passes):
                o = model(imgs, mask_indices=None)
                mc_probs.append(torch.sigmoid(o["pred_cls"]).cpu().numpy().flatten())
        mc_probs = np.stack(mc_probs)             # (T, B)
        uncertainties.extend(np.var(mc_probs, axis=0).tolist())
        model.eval()

    # ─── ECE & Reliability Diagram ───
    ece, bin_acc, bin_conf = calculate_ece(y_true_all, y_prob_all, n_bins=10)
    print(f"\n📈 ECE: {ece:.4f}")

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Mükemmel kalibrasyon")
    plt.plot(bin_conf, bin_acc, "o-", lw=2, color="darkorange",
             label=f"A-YOLO (ECE={ece:.4f})")
    plt.xlabel("Güven (predicted prob)")
    plt.ylabel("Gerçek doğruluk")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "calibration_curve.png"), dpi=150)
    plt.close()

    # ─── Uncertainty Histogram ───
    plt.figure(figsize=(8, 5))
    plt.hist(uncertainties, bins=50, color="purple", alpha=0.75)
    plt.xlabel("Epistemik belirsizlik (varyans)")
    plt.ylabel("Örnek sayısı")
    plt.title(f"MC Dropout ({args.mc_passes} pass) — Belirsizlik Dağılımı")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "uncertainty_distribution.png"), dpi=150)
    plt.close()

    # ─── Error Taxonomy ───
    total = sum(err.values())
    print("\n══════════════════════════════════════════════════")
    print("        🚀 ERROR TAXONOMY")
    print("══════════════════════════════════════════════════")
    for k, v in err.items():
        pct = 100 * v / total if total else 0
        print(f"  {k:<22}: {v:<6} ({pct:.1f}%)")
    print("══════════════════════════════════════════════════")

    # ─── Save artefacts ───
    pd.DataFrame([err]).to_csv(
        os.path.join(args.output_dir, "error_taxonomy.csv"), index=False)
    pd.DataFrame({
        "y_true": y_true_all,
        "y_prob": y_prob_all,
        "uncertainty": uncertainties,
    }).to_csv(os.path.join(args.output_dir, "calibration_data.csv"), index=False)

    summary = {
        "ece": ece,
        "mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
        "median_uncertainty": float(np.median(uncertainties)) if uncertainties else 0.0,
        **err,
    }
    with open(os.path.join(args.output_dir, "advanced_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 Sonuçlar: {args.output_dir}")


if __name__ == "__main__":
    main()