# Wrapper entrypoint for alpha search and fusion metrics
#s
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classifier olasılığı + GAN anomaly score + Grad-CAM score gibi farklı sinyalleri birleştirir.

Amaç:
- Tekil skorları normalize etmek
- Ağırlıklı fusion risk skoru üretmek
- Label varsa ROC-AUC / F1 / accuracy hesaplamak
- Opsiyonel olarak heatmap fusion yapmak

Desteklenen tipik kolon adları:
- classifier CSV: patient_id, probability
- GAN CSV: patient_id, gan_score / anomaly_score / mean_score / max_score
- CAM CSV: patient_id, cam_score / mean_score / max_score / pointing_hit

Örnek:
python scripts/eval/run_fusion_analysis.py \
    --classifier-csv outputs/eval/classifier_test/predictions.csv \
    --gan-csv outputs/eval/localization_gan/localization_per_image.csv \
    --cam-csv outputs/eval/localization_cam/localization_per_image.csv \
    --manifest data/splits/test_classifier.csv \
    --output-dir outputs/eval/fusion
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse classifier + GAN + CAM outputs.")
    parser.add_argument("--classifier-csv", type=str, required=True)
    parser.add_argument("--gan-csv", type=str, default=None)
    parser.add_argument("--cam-csv", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--w-cls", type=float, default=0.60)
    parser.add_argument("--w-gan", type=float, default=0.25)
    parser.add_argument("--w-cam", type=float, default=0.15)
    parser.add_argument("--search-threshold", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gan-map-dir", type=str, default=None)
    parser.add_argument("--cam-map-dir", type=str, default=None)
    parser.add_argument("--save-fused-maps", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-fused-maps", type=int, default=100)
    return parser.parse_args()


def minmax_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def choose_score_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Uygun skor kolonu bulunamadı. Mevcut kolonlar: {list(df.columns)}")


def read_map(path: Path, image_size: int) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze()
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return None
        arr = arr.astype(np.float32)
    arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def find_map(root: Optional[str], patient_id: str) -> Optional[Path]:
    if root is None:
        return None
    root_path = Path(root)
    for ext in [".png", ".jpg", ".jpeg", ".npy"]:
        p = root_path / f"{patient_id}{ext}"
        if p.exists():
            return p
    for suffix in ["_anomaly", "_gradcam", "_heatmap"]:
        for ext in [".png", ".jpg", ".jpeg", ".npy"]:
            p = root_path / f"{patient_id}{suffix}{ext}"
            if p.exists():
                return p
    return None


def search_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cls_df = pd.read_csv(args.classifier_csv)
    if "patient_id" not in cls_df.columns:
        raise ValueError("classifier csv içinde patient_id olmalı")

    cls_score_col = choose_score_column(cls_df, ["probability", "prob", "score"])
    merged = cls_df[["patient_id", cls_score_col]].rename(columns={cls_score_col: "cls_score"})

    if args.gan_csv is not None:
        gan_df = pd.read_csv(args.gan_csv)
        gan_score_col = choose_score_column(gan_df, ["gan_score", "anomaly_score", "mean_score", "max_score"])
        gan_df = gan_df[["patient_id", gan_score_col]].rename(columns={gan_score_col: "gan_score"})
        merged = merged.merge(gan_df, on="patient_id", how="left")
    else:
        merged["gan_score"] = 0.0

    if args.cam_csv is not None:
        cam_df = pd.read_csv(args.cam_csv)
        cam_score_col = choose_score_column(cam_df, ["cam_score", "mean_score", "max_score", "pointing_hit"])
        cam_df = cam_df[["patient_id", cam_score_col]].rename(columns={cam_score_col: "cam_score"})
        merged = merged.merge(cam_df, on="patient_id", how="left")
    else:
        merged["cam_score"] = 0.0

    merged["gan_score"] = merged["gan_score"].fillna(0.0)
    merged["cam_score"] = merged["cam_score"].fillna(0.0)

    merged["cls_score_n"] = minmax_norm(merged["cls_score"].values)
    merged["gan_score_n"] = minmax_norm(merged["gan_score"].values)
    merged["cam_score_n"] = minmax_norm(merged["cam_score"].values)

    merged["fusion_score"] = (
        args.w_cls * merged["cls_score_n"]
        + args.w_gan * merged["gan_score_n"]
        + args.w_cam * merged["cam_score_n"]
    )

    summary = {
        "weights": {
            "classifier": args.w_cls,
            "gan": args.w_gan,
            "cam": args.w_cam,
        }
    }

    if args.manifest is not None:
        manifest_df = pd.read_csv(args.manifest)[["patient_id", "label"]]
        merged = merged.merge(manifest_df, on="patient_id", how="left")

        valid = merged.dropna(subset=["label"]).copy()
        valid["label"] = valid["label"].astype(int)

        if args.search_threshold:
            used_thr = search_best_threshold(valid["label"].values, valid["fusion_score"].values)
        else:
            used_thr = args.threshold

        pred = (valid["fusion_score"].values >= used_thr).astype(int)
        y_true = valid["label"].values
        y_prob = valid["fusion_score"].values

        summary.update(
            {
                "threshold": float(used_thr),
                "accuracy": float(accuracy_score(y_true, pred)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
                "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
            }
        )
        merged["fusion_pred"] = (merged["fusion_score"] >= used_thr).astype(int)

    merged.to_csv(output_dir / "fusion_scores.csv", index=False)

    if args.save_fused_maps and args.gan_map_dir and args.cam_map_dir:
        fused_dir = output_dir / "fused_maps"
        fused_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for _, row in merged.iterrows():
            if count >= args.max_fused_maps:
                break
            patient_id = row["patient_id"]
            gan_path = find_map(args.gan_map_dir, patient_id)
            cam_path = find_map(args.cam_map_dir, patient_id)

            if gan_path is None and cam_path is None:
                continue

            gan_map = read_map(gan_path, args.image_size) if gan_path is not None else np.zeros((args.image_size, args.image_size), dtype=np.float32)
            cam_map = read_map(cam_path, args.image_size) if cam_path is not None else np.zeros((args.image_size, args.image_size), dtype=np.float32)

            fused = args.w_gan * gan_map + args.w_cam * cam_map
            fused = fused - fused.min()
            if fused.max() > 0:
                fused = fused / fused.max()

            fused_uint8 = (fused * 255).astype(np.uint8)
            cv2.imwrite(str(fused_dir / f"{patient_id}.png"), fused_uint8)
            count += 1

        summary["num_fused_maps"] = count

    with open(output_dir / "fusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Fusion analysis tamamlandı.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()