# Wrapper entrypoint for DICOM to PNG conversion
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#s
"""
RSNA Pneumonia DICOM görüntülerini PNG formatına çevirir.

Özellikler:
- DICOM okuma
- Modality LUT + VOI LUT uygulama
- MONOCHROME1 inversion düzeltme
- Percentile tabanlı normalize etme
- Opsiyonel CLAHE
- Yeniden boyutlandırma
- Çoklu iş parçacığı ile hızlı dönüştürme
- Metadata CSV üretme

Örnek:
python scripts/data/convert_dicom_to_png.py \
    --dicom-dir data/rsna/dicom \
    --output-dir data/images_png \
    --metadata-csv data/rsna/metadata.csv \
    --size 512 \
    --clahe
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RSNA DICOM images to PNG.")
    parser.add_argument("--dicom-dir", type=str, required=True, help="DICOM klasörü")
    parser.add_argument("--output-dir", type=str, required=True, help="PNG çıktı klasörü")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Çıktı metadata CSV yolu")
    parser.add_argument("--size", type=int, default=512, help="PNG yeniden boyutlandırma")
    parser.add_argument("--clahe", action="store_true", help="CLAHE uygula")
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--overwrite", action="store_true", help="Var olan PNG dosyalarını ez")
    parser.add_argument("--save-json-summary", type=str, default=None, help="Opsiyonel özet JSON yolu")
    return parser.parse_args()


def list_dicom_files(dicom_dir: str) -> List[Path]:
    root = Path(dicom_dir)
    files = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm"])
    return files


def safe_get(ds, key: str, default=None):
    return getattr(ds, key, default)


def dicom_to_uint8(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    DICOM -> uint8 grayscale.
    """
    img = ds.pixel_array.astype(np.float32)

    try:
        img = apply_modality_lut(img, ds).astype(np.float32)
    except Exception:
        pass

    try:
        img = apply_voi_lut(img, ds).astype(np.float32)
    except Exception:
        pass

    photometric = str(safe_get(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if photometric == "MONOCHROME1":
        img = img.max() - img

    # robust percentile normalization
    p1, p99 = np.percentile(img, (1, 99))
    if p99 <= p1:
        p1 = float(img.min())
        p99 = float(img.max()) if float(img.max()) > float(img.min()) else float(img.min()) + 1.0

    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)

    return img


def apply_clahe_uint8(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def process_one(
    dicom_path: Path,
    output_dir: Path,
    size: int,
    use_clahe: bool,
    overwrite: bool,
) -> Dict:
    patient_id = dicom_path.stem
    out_path = output_dir / f"{patient_id}.png"

    if out_path.exists() and not overwrite:
        return {
            "patient_id": patient_id,
            "dicom_path": str(dicom_path),
            "png_path": str(out_path),
            "status": "skipped_exists",
            "height": None,
            "width": None,
            "sex": None,
            "age": None,
            "view_position": None,
        }

    ds = pydicom.dcmread(str(dicom_path))
    img = dicom_to_uint8(ds)

    original_h, original_w = img.shape[:2]

    if use_clahe:
        img = apply_clahe_uint8(img)

    img = resize_image(img, size)
    cv2.imwrite(str(out_path), img)

    age_raw = safe_get(ds, "PatientAge", None)
    age_val = None
    if age_raw is not None:
        try:
            age_val = str(age_raw)
        except Exception:
            age_val = None

    row = {
        "patient_id": patient_id,
        "dicom_path": str(dicom_path),
        "png_path": str(out_path),
        "status": "ok",
        "height": int(original_h),
        "width": int(original_w),
        "sex": safe_get(ds, "PatientSex", None),
        "age": age_val,
        "view_position": safe_get(ds, "ViewPosition", None),
        "modality": safe_get(ds, "Modality", None),
        "photometric": safe_get(ds, "PhotometricInterpretation", None),
    }
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_files = list_dicom_files(args.dicom_dir)
    if len(dicom_files) == 0:
        raise FileNotFoundError(f"DICOM bulunamadı: {args.dicom_dir}")

    rows: List[Dict] = []
    failed: List[Dict] = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_one,
                dicom_path,
                output_dir,
                args.size,
                args.clahe,
                args.overwrite,
            )
            for dicom_path in dicom_files
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting DICOM -> PNG"):
            try:
                rows.append(future.result())
            except Exception as e:
                failed.append({"error": str(e)})

    df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)

    if args.metadata_csv is not None:
        Path(args.metadata_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.metadata_csv, index=False)

    if args.save_json_summary is not None:
        summary = {
            "num_dicoms": len(dicom_files),
            "num_success_rows": int((df["status"] == "ok").sum()) if "status" in df.columns else len(df),
            "num_skipped": int((df["status"] == "skipped_exists").sum()) if "status" in df.columns else 0,
            "num_failed": len(failed),
            "output_dir": str(output_dir),
            "size": args.size,
            "clahe": bool(args.clahe),
        }
        Path(args.save_json_summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] PNG dönüşümü tamamlandı: {output_dir}")
    if args.metadata_csv is not None:
        print(f"[OK] Metadata CSV: {args.metadata_csv}")
    if len(failed) > 0:
        print(f"[WARN] Hatalı dosya sayısı: {len(failed)}")


if __name__ == "__main__":
    main()