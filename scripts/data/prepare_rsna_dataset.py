# Wrapper entrypoint for dataset split and CSV preparation
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#s
"""
RSNA Pneumonia Detection Challenge için veri hazırlama scripti.

İşlevler:
- stage_2_train_labels.csv dosyasını okur
- patient bazında bbox'ları birleştirir
- PNG'lerle eşleştirir
- train / val / test split üretir
- classifier için manifest CSV üretir
- GAN eğitimi için sadece normal örneklerden ayrı manifest üretir
- class weights ve genel özet kaydeder

Örnek:
python scripts/data/prepare_rsna_dataset.py \
    --labels-csv data/rsna/stage_2_train_labels.csv \
    --png-dir data/images_png \
    --output-dir data/splits \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RSNA manifests and train/val/test splits.")
    parser.add_argument("--labels-csv", type=str, required=True, help="stage_2_train_labels.csv")
    parser.add_argument("--png-dir", type=str, required=True, help="PNG klasörü")
    parser.add_argument("--output-dir", type=str, required=True, help="Çıktı split klasörü")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-existing-png", action="store_true", help="PNG olmayan örnekleri at")
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train+val+test toplamı 1.0 olmalı. Mevcut toplam={total}")


def parse_rsna_labels(labels_csv: str, png_dir: str, require_existing_png: bool = False) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)

    required_cols = {"patientId", "x", "y", "width", "height", "Target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV içinde eksik kolonlar var: {missing}")

    grouped_rows: List[Dict] = []
    png_root = Path(png_dir)

    for patient_id, g in df.groupby("patientId"):
        g = g.reset_index(drop=True)

        label = int(g["Target"].max())
        boxes = []

        if label == 1:
            for _, row in g.iterrows():
                if int(row["Target"]) == 1:
                    boxes.append(
                        {
                            "x": float(row["x"]),
                            "y": float(row["y"]),
                            "width": float(row["width"]),
                            "height": float(row["height"]),
                        }
                    )

        png_path = png_root / f"{patient_id}.png"

        if require_existing_png and not png_path.exists():
            continue

        grouped_rows.append(
            {
                "patient_id": patient_id,
                "image_path": str(png_path),
                "label": label,
                "has_box": int(len(boxes) > 0),
                "bbox_count": len(boxes),
                "bboxes": json.dumps(boxes, ensure_ascii=False),
            }
        )

    out_df = pd.DataFrame(grouped_rows).sort_values("patient_id").reset_index(drop=True)
    return out_df


def stratified_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    y = df["label"].values

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=y,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["label"].values,
    )

    return (
        train_df.sort_values("patient_id").reset_index(drop=True),
        val_df.sort_values("patient_id").reset_index(drop=True),
        test_df.sort_values("patient_id").reset_index(drop=True),
    )


def add_basic_columns(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy()
    df["split"] = split_name
    df["exists"] = df["image_path"].apply(lambda p: int(Path(p).exists()))
    df["class_name"] = df["label"].map({0: "normal", 1: "pneumonia"})
    return df


def save_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def compute_class_weights(train_df: pd.DataFrame) -> Dict[str, float]:
    counts = train_df["label"].value_counts().to_dict()
    n0 = counts.get(0, 1)
    n1 = counts.get(1, 1)
    total = n0 + n1

    # BCE pos_weight için yaklaşık değer
    pos_weight = float(n0 / max(n1, 1))

    return {
        "num_negative": int(n0),
        "num_positive": int(n1),
        "total": int(total),
        "pos_weight": pos_weight,
    }


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = parse_rsna_labels(
        labels_csv=args.labels_csv,
        png_dir=args.png_dir,
        require_existing_png=args.require_existing_png,
    )

    if len(df) == 0:
        raise RuntimeError("Hazırlanan manifest boş. PNG yolu veya CSV içeriğini kontrol et.")

    train_df, val_df, test_df = stratified_split(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_df = add_basic_columns(train_df, "train")
    val_df = add_basic_columns(val_df, "val")
    test_df = add_basic_columns(test_df, "test")

    full_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

    save_manifest(full_df, output_dir / "all_manifest.csv")
    save_manifest(train_df, output_dir / "train.csv")
    save_manifest(val_df, output_dir / "val.csv")
    save_manifest(test_df, output_dir / "test.csv")

    # classifier manifests
    save_manifest(train_df[["patient_id", "image_path", "label", "class_name", "bboxes", "bbox_count", "split", "exists"]], output_dir / "train_classifier.csv")
    save_manifest(val_df[["patient_id", "image_path", "label", "class_name", "bboxes", "bbox_count", "split", "exists"]], output_dir / "val_classifier.csv")
    save_manifest(test_df[["patient_id", "image_path", "label", "class_name", "bboxes", "bbox_count", "split", "exists"]], output_dir / "test_classifier.csv")

    # GAN için normal örnekler
    train_gan_df = train_df[train_df["label"] == 0].reset_index(drop=True)
    val_gan_df = val_df[val_df["label"] == 0].reset_index(drop=True)
    save_manifest(train_gan_df, output_dir / "train_gan_normal.csv")
    save_manifest(val_gan_df, output_dir / "val_gan_normal.csv")

    # Pozitif test seti (lokalizasyon değerlendirmesi için yararlı)
    test_positive_df = test_df[test_df["label"] == 1].reset_index(drop=True)
    save_manifest(test_positive_df, output_dir / "test_positive.csv")

    class_weights = compute_class_weights(train_df)
    with open(output_dir / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2, ensure_ascii=False)

    summary = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "total": int(len(full_df)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "train_positive": int((train_df["label"] == 1).sum()),
        "train_negative": int((train_df["label"] == 0).sum()),
        "val_positive": int((val_df["label"] == 1).sum()),
        "val_negative": int((val_df["label"] == 0).sum()),
        "test_positive": int((test_df["label"] == 1).sum()),
        "test_negative": int((test_df["label"] == 0).sum()),
        "png_exists_ratio": float(full_df["exists"].mean()),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Veri hazırlama tamamlandı.")
    print(f"[OK] Çıktı klasörü: {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()