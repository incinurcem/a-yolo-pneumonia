import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pydicom


# ============================================================
# USER PATHS
# ============================================================

TRAIN_IMG_DIR = "/content/drive/MyDrive/Spring Semester/dataset/stage_2_train_images"
TRAIN_LABELS_CSV = "/content/drive/MyDrive/Spring Semester/dataset/stage_2_train_labels.csv"
CLASS_INFO_CSV = "/content/drive/MyDrive/Spring Semester/dataset/stage_2_detailed_class_info.csv"
DATASET_DIR = "/content/drive/MyDrive/Spring Semester/dataset"
TEST_IMG_DIR = "/content/drive/MyDrive/Spring Semester/dataset/stage_2_test_images"
METADATA_CLEAN_CSV = "/content/drive/MyDrive/Spring Semester/dataset/metadata_clean.csv"
SAMPLE_SUBMISSION_CSV = "/content/drive/MyDrive/Spring Semester/dataset/stage_2_sample_submission.csv"

OUTPUT_ROOT = os.path.join(DATASET_DIR, "prepared_rsna_detection_full")


# ============================================================
# SETTINGS
# ============================================================

VAL_SIZE = 0.15
RANDOM_STATE = 42
CATEGORY_ID = 1
CATEGORY_NAME = "pneumonia"


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = -1) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def load_csv_checked(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path)
    print(f"[OK] Loaded {name}: {path} | shape={df.shape}")
    return df


def get_dicom_path(image_dir: str, patient_id: str) -> str:
    return os.path.join(image_dir, f"{patient_id}.dcm")


# ============================================================
# DICOM HELPERS
# ============================================================

def read_dicom_metadata(dicom_path: str) -> Dict[str, Any]:
    """
    Reads basic DICOM metadata without loading pixel array.
    """
    result = {
        "dicom_ok": False,
        "height": -1,
        "width": -1,
        "view_position": "",
        "modality": "",
        "photometric_interpretation": "",
        "pixel_spacing_row": np.nan,
        "pixel_spacing_col": np.nan,
    }

    if not os.path.exists(dicom_path):
        return result

    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

        result["dicom_ok"] = True
        result["height"] = safe_int(getattr(ds, "Rows", -1), -1)
        result["width"] = safe_int(getattr(ds, "Columns", -1), -1)
        result["view_position"] = str(getattr(ds, "ViewPosition", ""))
        result["modality"] = str(getattr(ds, "Modality", ""))
        result["photometric_interpretation"] = str(getattr(ds, "PhotometricInterpretation", ""))

        pixel_spacing = getattr(ds, "PixelSpacing", None)
        if pixel_spacing is not None and len(pixel_spacing) >= 2:
            result["pixel_spacing_row"] = safe_float(pixel_spacing[0], np.nan)
            result["pixel_spacing_col"] = safe_float(pixel_spacing[1], np.nan)

    except Exception:
        pass

    return result


# ============================================================
# BBOX HELPERS
# ============================================================

def clean_and_validate_box(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int
) -> Tuple[bool, List[float], Dict[str, Any]]:
    """
    Keeps bbox in COCO xywh format.
    Clips it to image borders if size is known.
    """
    flags = {
        "was_clipped": False,
        "invalid_reason": ""
    }

    if np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h):
        flags["invalid_reason"] = "nan_box"
        return False, [], flags

    if w <= 0 or h <= 0:
        flags["invalid_reason"] = "non_positive_wh"
        return False, [], flags

    # If image size is unknown, keep the box if numerically valid
    if img_w <= 0 or img_h <= 0:
        return True, [float(x), float(y), float(w), float(h)], flags

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)

    new_w = x2 - x1
    new_h = y2 - y1

    if new_w <= 1 or new_h <= 1:
        flags["invalid_reason"] = "outside_or_too_small_after_clip"
        return False, [], flags

    if x1 != x or y1 != y or new_w != w or new_h != h:
        flags["was_clipped"] = True

    return True, [float(x1), float(y1), float(new_w), float(new_h)], flags


def extract_valid_boxes(
    group: pd.DataFrame,
    img_w: int,
    img_h: int
) -> Tuple[List[List[float]], int]:
    """
    Returns valid positive boxes in xywh format.
    """
    boxes = []
    clipped_count = 0

    for _, row in group.iterrows():
        target = safe_int(row.get("Target", 0), 0)
        if target != 1:
            continue

        x = safe_float(row.get("x", np.nan))
        y = safe_float(row.get("y", np.nan))
        w = safe_float(row.get("width", np.nan))
        h = safe_float(row.get("height", np.nan))

        is_valid, cleaned_box, flags = clean_and_validate_box(x, y, w, h, img_w, img_h)
        if is_valid:
            boxes.append(cleaned_box)
            if flags["was_clipped"]:
                clipped_count += 1

    return boxes, clipped_count


# ============================================================
# TRAIN AGGREGATION
# ============================================================

def aggregate_train_records(
    labels_df: pd.DataFrame,
    class_info_df: pd.DataFrame,
    train_img_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates one row per patientId.
    """
    required_label_cols = {"patientId", "Target", "x", "y", "width", "height"}
    required_class_cols = {"patientId", "class"}

    missing_label_cols = required_label_cols - set(labels_df.columns)
    missing_class_cols = required_class_cols - set(class_info_df.columns)

    if missing_label_cols:
        raise ValueError(f"TRAIN_LABELS_CSV missing columns: {missing_label_cols}")
    if missing_class_cols:
        raise ValueError(f"CLASS_INFO_CSV missing columns: {missing_class_cols}")

    merged = labels_df.merge(
        class_info_df[["patientId", "class"]],
        on="patientId",
        how="left"
    )

    grouped = merged.groupby("patientId", sort=True)

    records = []
    missing_files = []

    for patient_id, group in tqdm(grouped, total=len(grouped), desc="Aggregating train records"):
        dicom_path = get_dicom_path(train_img_dir, patient_id)
        dicom_exists = os.path.exists(dicom_path)

        if not dicom_exists:
            missing_files.append({
                "patientId": patient_id,
                "split": "train",
                "expected_path": dicom_path
            })
            dicom_meta = {
                "dicom_ok": False,
                "height": -1,
                "width": -1,
                "view_position": "",
                "modality": "",
                "photometric_interpretation": "",
                "pixel_spacing_row": np.nan,
                "pixel_spacing_col": np.nan,
            }
        else:
            dicom_meta = read_dicom_metadata(dicom_path)

        img_h = dicom_meta["height"]
        img_w = dicom_meta["width"]

        boxes, clipped_count = extract_valid_boxes(group, img_w=img_w, img_h=img_h)
        target = 1 if len(boxes) > 0 else 0

        class_values = group["class"].dropna().unique().tolist()
        class_name = class_values[0] if len(class_values) > 0 else "Unknown"

        records.append({
            "patientId": patient_id,
            "dicom_path": dicom_path if dicom_exists else "",
            "dicom_exists": int(dicom_exists),
            "dicom_ok": int(dicom_meta["dicom_ok"]),
            "height": int(dicom_meta["height"]),
            "width": int(dicom_meta["width"]),
            "view_position": dicom_meta["view_position"],
            "modality": dicom_meta["modality"],
            "photometric_interpretation": dicom_meta["photometric_interpretation"],
            "pixel_spacing_row": dicom_meta["pixel_spacing_row"],
            "pixel_spacing_col": dicom_meta["pixel_spacing_col"],
            "target": int(target),
            "num_boxes": int(len(boxes)),
            "num_clipped_boxes": int(clipped_count),
            "boxes_xywh": json.dumps(boxes),
            "class_name": class_name
        })

    master_df = pd.DataFrame(records)
    missing_files_df = pd.DataFrame(missing_files)

    return master_df, missing_files_df


# ============================================================
# TEST AGGREGATION
# ============================================================

def build_test_records(
    test_img_dir: str,
    sample_submission_csv: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_df = load_csv_checked(sample_submission_csv, "SAMPLE_SUBMISSION_CSV")

    if "patientId" not in sample_df.columns:
        raise ValueError("SAMPLE_SUBMISSION_CSV must contain column 'patientId'")

    records = []
    missing_files = []

    for patient_id in tqdm(sample_df["patientId"].tolist(), desc="Building test records"):
        dicom_path = get_dicom_path(test_img_dir, patient_id)
        dicom_exists = os.path.exists(dicom_path)

        if not dicom_exists:
            missing_files.append({
                "patientId": patient_id,
                "split": "test",
                "expected_path": dicom_path
            })
            dicom_meta = {
                "dicom_ok": False,
                "height": -1,
                "width": -1,
                "view_position": "",
                "modality": "",
                "photometric_interpretation": "",
                "pixel_spacing_row": np.nan,
                "pixel_spacing_col": np.nan,
            }
        else:
            dicom_meta = read_dicom_metadata(dicom_path)

        records.append({
            "patientId": patient_id,
            "dicom_path": dicom_path if dicom_exists else "",
            "dicom_exists": int(dicom_exists),
            "dicom_ok": int(dicom_meta["dicom_ok"]),
            "height": int(dicom_meta["height"]),
            "width": int(dicom_meta["width"]),
            "view_position": dicom_meta["view_position"],
            "modality": dicom_meta["modality"],
            "photometric_interpretation": dicom_meta["photometric_interpretation"],
            "pixel_spacing_row": dicom_meta["pixel_spacing_row"],
            "pixel_spacing_col": dicom_meta["pixel_spacing_col"]
        })

    test_df = pd.DataFrame(records)
    missing_files_df = pd.DataFrame(missing_files)
    return test_df, missing_files_df


# ============================================================
# SPLIT
# ============================================================

def split_master_df(
    master_df: pd.DataFrame,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "target" not in master_df.columns:
        raise ValueError("master_df must contain 'target'")

    train_df, val_df = train_test_split(
        master_df,
        test_size=val_size,
        random_state=random_state,
        stratify=master_df["target"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ============================================================
# COCO CONVERSION
# ============================================================

def build_coco_json(
    split_df: pd.DataFrame,
    output_json_path: str,
    category_id: int = 1,
    category_name: str = "pneumonia"
) -> Dict[str, Any]:
    images = []
    annotations = []
    categories = [{"id": category_id, "name": category_name}]
    ann_id = 1

    for image_id, row in enumerate(split_df.itertuples(index=False), start=1):
        patient_id = row.patientId
        dicom_path = row.dicom_path
        boxes = json.loads(row.boxes_xywh)

        images.append({
            "id": image_id,
            "file_name": dicom_path,
            "patientId": patient_id,
            "width": int(row.width),
            "height": int(row.height)
        })

        for box in boxes:
            x, y, w, h = box
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    return coco


# ============================================================
# SUMMARY
# ============================================================

def summarize_split(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    summary = {
        "name": name,
        "num_images": int(len(df)),
    }

    if "target" in df.columns:
        summary["num_positive_images"] = int((df["target"] == 1).sum())
        summary["num_negative_images"] = int((df["target"] == 0).sum())
        summary["positive_ratio"] = float((df["target"] == 1).mean())
        summary["num_total_boxes"] = int(df["num_boxes"].sum())
        summary["avg_boxes_per_positive_image"] = (
            float(df.loc[df["target"] == 1, "num_boxes"].mean())
            if (df["target"] == 1).sum() > 0 else 0.0
        )

    if "dicom_exists" in df.columns:
        summary["num_missing_dicom"] = int((df["dicom_exists"] == 0).sum())

    if "dicom_ok" in df.columns:
        summary["num_corrupted_or_unreadable_dicom"] = int((df["dicom_ok"] == 0).sum())

    return summary


def save_summary(summary_dict: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
    metadata_dir = os.path.join(OUTPUT_ROOT, "metadata")
    ann_dir = os.path.join(OUTPUT_ROOT, "annotations")
    reports_dir = os.path.join(OUTPUT_ROOT, "reports")

    ensure_dir(OUTPUT_ROOT)
    ensure_dir(metadata_dir)
    ensure_dir(ann_dir)
    ensure_dir(reports_dir)

    # --------------------------------------------------------
    # Load CSVs
    # --------------------------------------------------------
    labels_df = load_csv_checked(TRAIN_LABELS_CSV, "TRAIN_LABELS_CSV")
    class_info_df = load_csv_checked(CLASS_INFO_CSV, "CLASS_INFO_CSV")

    if os.path.exists(METADATA_CLEAN_CSV):
        metadata_clean_df = load_csv_checked(METADATA_CLEAN_CSV, "METADATA_CLEAN_CSV")
        print(f"[INFO] metadata_clean.csv available: shape={metadata_clean_df.shape}")
    else:
        print("[WARN] METADATA_CLEAN_CSV not found. Continuing without it.")

    # --------------------------------------------------------
    # Aggregate train
    # --------------------------------------------------------
    all_train_master_df, missing_train_df = aggregate_train_records(
        labels_df=labels_df,
        class_info_df=class_info_df,
        train_img_dir=TRAIN_IMG_DIR
    )

    # --------------------------------------------------------
    # Aggregate test
    # --------------------------------------------------------
    test_master_df, missing_test_df = build_test_records(
        test_img_dir=TEST_IMG_DIR,
        sample_submission_csv=SAMPLE_SUBMISSION_CSV
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------
    train_df, val_df = split_master_df(
        master_df=all_train_master_df,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Save CSVs
    # --------------------------------------------------------
    all_train_master_csv = os.path.join(metadata_dir, "all_train_master.csv")
    train_master_csv = os.path.join(metadata_dir, "train_master.csv")
    val_master_csv = os.path.join(metadata_dir, "val_master.csv")
    test_master_csv = os.path.join(metadata_dir, "test_master.csv")
    train_positive_only_csv = os.path.join(metadata_dir, "train_positive_only.csv")
    val_positive_only_csv = os.path.join(metadata_dir, "val_positive_only.csv")

    all_train_master_df.to_csv(all_train_master_csv, index=False)
    train_df.to_csv(train_master_csv, index=False)
    val_df.to_csv(val_master_csv, index=False)
    test_master_df.to_csv(test_master_csv, index=False)

    train_df[train_df["target"] == 1].to_csv(train_positive_only_csv, index=False)
    val_df[val_df["target"] == 1].to_csv(val_positive_only_csv, index=False)

    # --------------------------------------------------------
    # Save missing files report
    # --------------------------------------------------------
    missing_files_df = pd.concat([missing_train_df, missing_test_df], ignore_index=True)
    missing_files_csv = os.path.join(reports_dir, "missing_files.csv")
    missing_files_df.to_csv(missing_files_csv, index=False)

    # --------------------------------------------------------
    # Build COCO JSON
    # --------------------------------------------------------
    train_coco_json = os.path.join(ann_dir, "train_coco.json")
    val_coco_json = os.path.join(ann_dir, "val_coco.json")

    train_coco = build_coco_json(
        split_df=train_df,
        output_json_path=train_coco_json,
        category_id=CATEGORY_ID,
        category_name=CATEGORY_NAME
    )

    val_coco = build_coco_json(
        split_df=val_df,
        output_json_path=val_coco_json,
        category_id=CATEGORY_ID,
        category_name=CATEGORY_NAME
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary = {
        "source_paths": {
            "TRAIN_IMG_DIR": TRAIN_IMG_DIR,
            "TRAIN_LABELS_CSV": TRAIN_LABELS_CSV,
            "CLASS_INFO_CSV": CLASS_INFO_CSV,
            "TEST_IMG_DIR": TEST_IMG_DIR,
            "METADATA_CLEAN_CSV": METADATA_CLEAN_CSV,
            "SAMPLE_SUBMISSION_CSV": SAMPLE_SUBMISSION_CSV
        },
        "output_paths": {
            "OUTPUT_ROOT": OUTPUT_ROOT,
            "all_train_master_csv": all_train_master_csv,
            "train_master_csv": train_master_csv,
            "val_master_csv": val_master_csv,
            "test_master_csv": test_master_csv,
            "train_coco_json": train_coco_json,
            "val_coco_json": val_coco_json,
            "missing_files_csv": missing_files_csv
        },
        "dataset_summary": {
            "all_train": summarize_split(all_train_master_df, "all_train"),
            "train": summarize_split(train_df, "train"),
            "val": summarize_split(val_df, "val"),
            "test": summarize_split(test_master_df, "test")
        },
        "annotation_summary": {
            "train_num_images": len(train_coco["images"]),
            "train_num_annotations": len(train_coco["annotations"]),
            "val_num_images": len(val_coco["images"]),
            "val_num_annotations": len(val_coco["annotations"])
        },
        "missing_files_count": int(len(missing_files_df))
    }

    summary_json = os.path.join(reports_dir, "summary.json")
    save_summary(summary, summary_json)

    print("\n" + "=" * 90)
    print("[DONE] RSNA detection preprocessing completed.")
    print(f"[INFO] Output root: {OUTPUT_ROOT}")
    print(f"[INFO] Train CSV: {train_master_csv}")
    print(f"[INFO] Val CSV:   {val_master_csv}")
    print(f"[INFO] Test CSV:  {test_master_csv}")
    print(f"[INFO] Train COCO: {train_coco_json}")
    print(f"[INFO] Val COCO:   {val_coco_json}")
    print(f"[INFO] Missing files CSV: {missing_files_csv}")
    print(f"[INFO] Summary JSON: {summary_json}")
    print("=" * 90)


if __name__ == "__main__":
    main()