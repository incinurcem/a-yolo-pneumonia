# Wrapper entrypoint for classifier metrics and plots
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#s
"""
Classifier checkpoint'ini test/val manifest üzerinde değerlendirir.

Metrikler:
- Accuracy
- Precision
- Recall
- Specificity
- F1
- ROC-AUC
- PR-AUC
- Brier score
- ECE
- Confusion matrix
- ROC / PR / calibration eğrileri

Örnek:
python scripts/eval/evaluate_classifier.py \
    --manifest data/splits/test_classifier.csv \
    --checkpoint outputs/classifier/best.pt \
    --output-dir outputs/eval/classifier_test \
    --batch-size 16 \
    --num-workers 4
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pneumonia classifier.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None, help="Checkpoint threshold yoksa kullanılır")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class PneumoniaDataset(Dataset):
    def __init__(self, manifest_path: str, image_size: int, mean=None, std=None):
        self.df = pd.read_csv(manifest_path)
        self.image_size = image_size
        self.mean = np.array(mean if mean is not None else [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array(std if std is not None else [0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"])
        patient_id = row["patient_id"]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {img_path}")

        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))

        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "patient_id": patient_id,
            "img_path": img_path,
        }


class PneumoniaClassifier(nn.Module):
    def __init__(self, model_name: str = "resnet50", pretrained: bool = False, in_chans: int = 3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,
            in_chans=in_chans,
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def load_checkpoint(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_name = ckpt.get("model_name", "resnet50")
    image_size = int(ckpt.get("image_size", 512))
    in_chans = int(ckpt.get("in_chans", 3))
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])
    threshold = ckpt.get("threshold", 0.5)

    model = PneumoniaClassifier(model_name=model_name, pretrained=False, in_chans=in_chans)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return model, image_size, mean, std, threshold, ckpt


@torch.no_grad()
def run_inference(model, loader, device: str):
    all_probs = []
    all_labels = []
    all_ids = []
    all_paths = []

    for batch in tqdm(loader, desc="Evaluating classifier"):
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy().astype(int)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(y.tolist())
        all_ids.extend(batch["patient_id"])
        all_paths.extend(batch["img_path"])

    all_probs = np.array(all_probs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    return {
        "probs": all_probs,
        "labels": all_labels,
        "patient_ids": all_ids,
        "img_paths": all_paths,
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_prob >= left) & (y_prob < right) if i < n_bins - 1 else (y_prob >= left) & (y_prob <= right)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / max(tn + fp, 1)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "Pneumonia"])
    plt.yticks(tick_marks, ["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    fig = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig = plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path, n_bins: int = 10) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = []
    bin_conf = []

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_prob >= left) & (y_prob < right) if i < n_bins - 1 else (y_prob >= left) & (y_prob <= right)
        if mask.sum() == 0:
            continue
        bin_conf.append(float(y_prob[mask].mean()))
        bin_acc.append(float(y_true[mask].mean()))

    fig = plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    if len(bin_conf) > 0:
        plt.plot(bin_conf, bin_acc, marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, image_size, mean, std, ckpt_thr, ckpt = load_checkpoint(args.checkpoint, args.device)
    threshold = args.threshold if args.threshold is not None else ckpt_thr

    dataset = PneumoniaDataset(args.manifest, image_size=image_size, mean=mean, std=std)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    result = run_inference(model, loader, args.device)
    y_true = result["labels"]
    y_prob = result["probs"]

    best_thr = find_best_threshold(y_true, y_prob)
    metrics_default = compute_metrics(y_true, y_prob, threshold)
    metrics_bestf1 = compute_metrics(y_true, y_prob, best_thr)

    pred_df = pd.DataFrame(
        {
            "patient_id": result["patient_ids"],
            "img_path": result["img_paths"],
            "label": y_true,
            "probability": y_prob,
            "pred_default": (y_prob >= threshold).astype(int),
            "pred_bestf1": (y_prob >= best_thr).astype(int),
        }
    )
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    report = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "image_size": image_size,
        "model_name": ckpt.get("model_name", "resnet50"),
        "metrics_default_threshold": metrics_default,
        "metrics_best_f1_threshold": metrics_bestf1,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1])
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, output_dir / "roc_curve.png")
    plot_pr_curve(y_true, y_prob, output_dir / "pr_curve.png")
    plot_calibration(y_true, y_prob, output_dir / "calibration_curve.png")

    print("[OK] Classifier evaluation tamamlandı.")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()