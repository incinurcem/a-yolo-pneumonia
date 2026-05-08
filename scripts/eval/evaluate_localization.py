# Wrapper entrypoint for anomaly-map localization evaluation
#s
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAN anomaly map / Grad-CAM / heatmap lokalizasyonunu RSNA bbox ground-truth ile değerlendirir.

Beklenen giriş:
- test manifest CSV (boxes_json kolonunu içermeli)
- heatmap klasörü (png/jpg/npy dosyaları)

Metrikler:
- IoU
- Dice
- Hit Rate
- Pointing Game
- Predicted area ratio
- Mean / max anomaly score

Örnek:
python scripts/eval/evaluate_localization.py \
    --manifest data/splits/test_positive.csv \
    --heatmap-dir outputs/infer/gan_maps \
    --output-dir outputs/eval/localization_gan \
    --threshold-mode percentile \
    --threshold-value 90
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate localization maps against RSNA bboxes.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--heatmap-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold-mode", type=str, default="percentile", choices=["percentile", "fixed"])
    parser.add_argument("--threshold-value", type=float, default=90.0)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--max-overlay", type=int, default=50)
    return parser.parse_args()


def read_heatmap(path: Path, image_size: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        if arr.ndim == 3:
            arr = arr.squeeze()
        arr = arr.astype(np.float32)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(f"Heatmap okunamadı: {path}")
        arr = arr.astype(np.float32)

    arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = arr - arr.min()
    denom = arr.max() - arr.min()
    if denom > 0:
        arr = arr / denom
    return arr.astype(np.float32)


def find_heatmap_file(heatmap_dir: Path, patient_id: str) -> Optional[Path]:
    candidates = [
        heatmap_dir / f"{patient_id}.png",
        heatmap_dir / f"{patient_id}.jpg",
        heatmap_dir / f"{patient_id}.jpeg",
        heatmap_dir / f"{patient_id}.npy",
        heatmap_dir / f"{patient_id}_heatmap.png",
        heatmap_dir / f"{patient_id}_anomaly.png",
        heatmap_dir / f"{patient_id}_gradcam.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_gt_mask(bboxes: str, image_size: int) -> np.ndarray:
    gt = np.zeros((image_size, image_size), dtype=np.uint8)
    boxes = json.loads(bboxes) if isinstance(bboxes, str) else []

    # RSNA bbox koordinatları orijinal boyutta; burada basit normalize kabulü yerine
    # çoğu pratik kullanım için 1024 tabanına göre ölçekliyoruz.
    src_size = 1024.0
    scale = image_size / src_size

    for box in boxes:
        x = int(round(float(box["x"]) * scale))
        y = int(round(float(box["y"]) * scale))
        w = int(round(float(box["width"]) * scale))
        h = int(round(float(box["height"]) * scale))

        x1 = np.clip(x, 0, image_size - 1)
        y1 = np.clip(y, 0, image_size - 1)
        x2 = np.clip(x + w, 0, image_size)
        y2 = np.clip(y + h, 0, image_size)

        gt[y1:y2, x1:x2] = 1
    return gt


def threshold_heatmap(heatmap: np.ndarray, mode: str, value: float) -> np.ndarray:
    if mode == "percentile":
        thr = np.percentile(heatmap, value)
    else:
        thr = value
    return (heatmap >= thr).astype(np.uint8)


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    iou = inter / (union + 1e-8)
    dice = 2.0 * inter / (pred_sum + gt_sum + 1e-8)
    hit = 1.0 if inter > 0 else 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "hit": float(hit),
        "pred_area_ratio": float(pred_sum / pred.size),
        "gt_area_ratio": float(gt_sum / gt.size),
    }


def pointing_game(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return float(gt_mask[y, x] > 0)


def save_overlay(heatmap: np.ndarray, gt_mask: np.ndarray, output_path: Path) -> None:
    hm = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    gt_color = np.zeros_like(hm_color)
    gt_color[:, :, 1] = (gt_mask * 255).astype(np.uint8)

    overlay = cv2.addWeighted(hm_color, 0.75, gt_color, 0.25, 0)
    cv2.imwrite(str(output_path), overlay)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = output_dir / "overlays"
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    if "bboxes" not in df.columns:
        raise ValueError("Manifest içinde bboxes kolonu olmalı.")

    rows = []
    saved_overlay_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating localization"):
        patient_id = row["patient_id"]
        heatmap_path = find_heatmap_file(Path(args.heatmap_dir), patient_id)
        if heatmap_path is None:
            rows.append(
                {
                    "patient_id": patient_id,
                    "found_heatmap": 0,
                    "iou": np.nan,
                    "dice": np.nan,
                    "hit": np.nan,
                    "pointing_hit": np.nan,
                    "mean_score": np.nan,
                    "max_score": np.nan,
                    "pred_area_ratio": np.nan,
                    "gt_area_ratio": np.nan,
                }
            )
            continue

        heatmap = read_heatmap(heatmap_path, args.image_size)
        gt_mask = build_gt_mask(row["bboxes"], args.image_size)
        pred_mask = threshold_heatmap(heatmap, args.threshold_mode, args.threshold_value)

        metrics = compute_binary_metrics(pred_mask, gt_mask)
        pgame = pointing_game(heatmap, gt_mask)

        result_row = {
            "patient_id": patient_id,
            "found_heatmap": 1,
            "heatmap_path": str(heatmap_path),
            "iou": metrics["iou"],
            "dice": metrics["dice"],
            "hit": metrics["hit"],
            "pointing_hit": pgame,
            "mean_score": float(heatmap.mean()),
            "max_score": float(heatmap.max()),
            "pred_area_ratio": metrics["pred_area_ratio"],
            "gt_area_ratio": metrics["gt_area_ratio"],
        }
        rows.append(result_row)

        if args.save_overlays and saved_overlay_count < args.max_overlay:
            save_overlay(heatmap, gt_mask, overlay_dir / f"{patient_id}_overlay.png")
            saved_overlay_count += 1

    res_df = pd.DataFrame(rows)
    res_df.to_csv(output_dir / "localization_per_image.csv", index=False)

    valid_df = res_df[res_df["found_heatmap"] == 1].dropna()

    summary = {
        "manifest": args.manifest,
        "heatmap_dir": args.heatmap_dir,
        "num_samples": int(len(df)),
        "num_found_heatmaps": int((res_df["found_heatmap"] == 1).sum()),
        "mean_iou": float(valid_df["iou"].mean()) if len(valid_df) > 0 else None,
        "mean_dice": float(valid_df["dice"].mean()) if len(valid_df) > 0 else None,
        "hit_rate": float(valid_df["hit"].mean()) if len(valid_df) > 0 else None,
        "pointing_game": float(valid_df["pointing_hit"].mean()) if len(valid_df) > 0 else None,
        "mean_pred_area_ratio": float(valid_df["pred_area_ratio"].mean()) if len(valid_df) > 0 else None,
    }

    with open(output_dir / "localization_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Lokalizasyon değerlendirmesi tamamlandı.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()