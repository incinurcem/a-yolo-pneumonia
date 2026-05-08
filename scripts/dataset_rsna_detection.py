import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


# ============================================================
# CONFIG
# ============================================================

def _constant_pad_kwargs():
    """
    Albumentations 1.x / 2.x uyumluluğu için
    sabit padding argümanlarını üretir.
    """
    return {
        "border_mode": cv2.BORDER_CONSTANT,
        "value": 0,          # PadIfNeeded için güvenli
        "mask_value": 0,     # mask varsa sorun çıkarmasın
    }


def _constant_warp_kwargs():
    """
    Geometric transforms için sabit border argümanları.
    """
    return {
        "border_mode": cv2.BORDER_CONSTANT,
        "value": 0,
    }

@dataclass
class RSNADetectionConfig:
    image_size: int = 384
    num_workers: int = 12   # Performans ve stabilite için 12 olarak güncellendi
    batch_size: int = 32
    max_pixel_value: float = 255.0

    # intensity preprocessing
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # normalization
    norm_mode: str = "imagenet"   # "imagenet" or "minmax_01"
    to_3channel: bool = False     # model input_adapter 1 kanal bekliyorsa False

    # augmentation
    train_horizontal_flip_p: float = 0.5
    train_shift_scale_rotate_p: float = 0.5
    train_shift_limit: float = 0.02
    train_scale_limit: float = 0.05
    train_rotate_limit: int = 7
    train_random_brightness_contrast_p: float = 0.3
    train_gaussian_blur_p: float = 0.1

    # bbox filtering
    min_bbox_visibility: float = 0.1
    min_area: float = 1.0


# ============================================================
# IMAGE READING
# ============================================================

def read_png_grayscale(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"PNG image not found: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    return image.astype(np.uint8)


def apply_clahe_if_needed(
    image_uint8: np.ndarray,
    enabled: bool,
    clip_limit: float,
    tile_grid_size: Tuple[int, int]
) -> np.ndarray:
    if not enabled:
        return image_uint8

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(image_uint8)


def grayscale_to_3channel(image_uint8: np.ndarray) -> np.ndarray:
    return np.stack([image_uint8, image_uint8, image_uint8], axis=-1)


# ============================================================
# BOX HELPERS
# ============================================================

def safe_parse_boxes_xywh(value) -> List[List[float]]:
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() == "nan":
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return []
        except Exception:
            return []

    return []


def xywh_to_pascal_voc(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def pascal_voc_to_cxcywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return [cx, cy, w, h]


def normalize_cxcywh(box: List[float], image_w: int, image_h: int) -> List[float]:
    cx, cy, w, h = box
    return [
        cx / image_w,
        cy / image_h,
        w / image_w,
        h / image_h,
    ]


def sanitize_pascal_boxes(
    boxes: List[List[float]],
    labels: List[int],
    image_w: int,
    image_h: int
) -> Tuple[List[List[float]], List[int]]:
    clean_boxes = []
    clean_labels = []

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box

        x1 = max(0.0, min(float(image_w - 1), float(x1)))
        y1 = max(0.0, min(float(image_h - 1), float(y1)))
        x2 = max(0.0, min(float(image_w), float(x2)))
        y2 = max(0.0, min(float(image_h), float(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        clean_boxes.append([x1, y1, x2, y2])
        clean_labels.append(int(label))

    return clean_boxes, clean_labels


# ============================================================
# TRANSFORMS
# ============================================================

def _build_normalize(cfg: RSNADetectionConfig):
    if cfg.to_3channel:
        if cfg.norm_mode == "imagenet":
            return A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=cfg.max_pixel_value
            )
        elif cfg.norm_mode == "minmax_01":
            return A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=cfg.max_pixel_value
            )
        else:
            raise ValueError(f"Unsupported norm_mode: {cfg.norm_mode}")
    else:
        if cfg.norm_mode == "imagenet":
            return A.Normalize(
                mean=(0.485,),
                std=(0.229,),
                max_pixel_value=cfg.max_pixel_value
            )
        elif cfg.norm_mode == "minmax_01":
            return A.Normalize(
                mean=(0.0,),
                std=(1.0,),
                max_pixel_value=cfg.max_pixel_value
            )
        else:
            raise ValueError(f"Unsupported norm_mode: {cfg.norm_mode}")


def get_train_transform(cfg: RSNADetectionConfig) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=cfg.image_size),

            A.PadIfNeeded(
                min_height=cfg.image_size,
                min_width=cfg.image_size,
                p=1.0,
                **_constant_pad_kwargs()
            ),

            A.HorizontalFlip(p=cfg.train_horizontal_flip_p),

            A.ShiftScaleRotate(
                shift_limit=cfg.train_shift_limit,
                scale_limit=cfg.train_scale_limit,
                rotate_limit=cfg.train_rotate_limit,
                interpolation=cv2.INTER_LINEAR,
                p=cfg.train_shift_scale_rotate_p,
                **_constant_warp_kwargs()
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=cfg.train_random_brightness_contrast_p
            ),

            A.GaussianBlur(
                blur_limit=(3, 5),
                p=cfg.train_gaussian_blur_p
            ),

            _build_normalize(cfg),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=cfg.min_bbox_visibility,
            min_area=cfg.min_area
        )
    )


def get_valid_transform(cfg: RSNADetectionConfig) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=cfg.image_size),

            A.PadIfNeeded(
                min_height=cfg.image_size,
                min_width=cfg.image_size,
                p=1.0,
                **_constant_pad_kwargs()
            ),

            _build_normalize(cfg),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0,
            min_area=0.0
        )
    )


def get_test_transform(cfg: RSNADetectionConfig) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=cfg.image_size),

            A.PadIfNeeded(
                min_height=cfg.image_size,
                min_width=cfg.image_size,
                p=1.0,
                **_constant_pad_kwargs()
            ),

            _build_normalize(cfg),
        ]
    )


# ============================================================
# TARGET BUILDER
# ============================================================

def build_detection_target(
    boxes_pascal: List[List[float]],
    labels: List[int],
    image_id: int,
    patient_id: str,
    resized_h: int,
    resized_w: int,
    orig_h: int,
    orig_w: int,
) -> Dict[str, Any]:
    if len(boxes_pascal) == 0:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,), dtype=torch.int64)
        area_tensor = torch.zeros((0,), dtype=torch.float32)
        iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes_cxcywh = []
        areas = []

        for box in boxes_pascal:
            x1, y1, x2, y2 = box
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            cxcywh = pascal_voc_to_cxcywh(box)
            cxcywh = normalize_cxcywh(cxcywh, resized_w, resized_h)

            boxes_cxcywh.append(cxcywh)
            areas.append(area)

        boxes_tensor = torch.tensor(boxes_cxcywh, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area_tensor = torch.tensor(areas, dtype=torch.float32)
        iscrowd_tensor = torch.zeros((len(boxes_pascal),), dtype=torch.int64)

    target = {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "area": area_tensor,
        "iscrowd": iscrowd_tensor,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "size": torch.tensor([resized_h, resized_w], dtype=torch.int64),
        "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
        "patient_id": patient_id,
    }
    return target


# ============================================================
# BASE DATASET
# ============================================================

class BaseRSNADetectionDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        transform: Optional[A.Compose],
        cfg: RSNADetectionConfig,
        is_test: bool = False
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.cfg = cfg
        self.is_test = is_test

        if "patientId" not in self.df.columns:
            raise ValueError(f"{csv_path} must contain 'patientId'")

        if "image_path" not in self.df.columns:
            raise ValueError(f"{csv_path} must contain 'image_path'")

        if not self.is_test:
            required_cols = ["target", "boxes_xywh"]
            for col in required_cols:
                if col not in self.df.columns:
                    raise ValueError(f"{csv_path} must contain '{col}'")

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str) -> np.ndarray:
        image = read_png_grayscale(image_path)

        image = apply_clahe_if_needed(
            image,
            enabled=self.cfg.apply_clahe,
            clip_limit=self.cfg.clahe_clip_limit,
            tile_grid_size=self.cfg.clahe_tile_grid_size
        )

        if self.cfg.to_3channel:
            image = grayscale_to_3channel(image)
        else:
            image = np.expand_dims(image, axis=-1)

        return image


# ============================================================
# TRAIN / VAL DATASET
# ============================================================

class RSNADetectionTrainDataset(BaseRSNADetectionDataset):
    def __init__(
        self,
        csv_path: str,
        transform: Optional[A.Compose],
        cfg: RSNADetectionConfig
    ):
        super().__init__(
            csv_path=csv_path,
            transform=transform,
            cfg=cfg,
            is_test=False
        )

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        patient_id = str(row["patientId"])
        image_path = str(row["image_path"])
        dicom_path = str(row.get("dicom_path", ""))

        if image_path == "" or not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image_path for patientId={patient_id}: {image_path}")

        image = self._load_image(image_path)
        orig_h, orig_w = image.shape[:2]

        boxes_xywh = safe_parse_boxes_xywh(row.get("boxes_xywh", "[]"))
        
        # --- KRİTİK GÜNCELLEME ---
        # 0: lesion (foreground), 1: background (no-object)
        # DETR ve Criterion standartlarına uyum için lezyon etiketi 0 yapıldı.
        labels = [0] * len(boxes_xywh) 

        boxes_pascal = [xywh_to_pascal_voc(box) for box in boxes_xywh]
        boxes_pascal, labels = sanitize_pascal_boxes(
            boxes_pascal,
            labels,
            image_w=orig_w,
            image_h=orig_h
        )

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes_pascal,
                labels=labels
            )
            image = transformed["image"]
            boxes_pascal = [list(b) for b in transformed["bboxes"]]
            labels = list(transformed["labels"])

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        resized_h, resized_w = image.shape[1], image.shape[2]

        target = build_detection_target(
            boxes_pascal=boxes_pascal,
            labels=labels,
            image_id=index,
            patient_id=patient_id,
            resized_h=resized_h,
            resized_w=resized_w,
            orig_h=orig_h,
            orig_w=orig_w,
        )

        meta = {
            "patient_id": patient_id,
            "image_path": image_path,
            "dicom_path": dicom_path,
            "orig_height": orig_h,
            "orig_width": orig_w,
            "resized_height": resized_h,
            "resized_width": resized_w,
            "target_binary": int(row.get("target", 0)),
        }

        return image, target, meta


# ============================================================
# TEST DATASET
# ============================================================

class RSNADetectionTestDataset(BaseRSNADetectionDataset):
    def __init__(
        self,
        csv_path: str,
        transform: Optional[A.Compose],
        cfg: RSNADetectionConfig
    ):
        super().__init__(
            csv_path=csv_path,
            transform=transform,
            cfg=cfg,
            is_test=True
        )

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        patient_id = str(row["patientId"])
        image_path = str(row["image_path"])
        dicom_path = str(row.get("dicom_path", ""))

        if image_path == "" or not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image_path for patientId={patient_id}: {image_path}")

        image = self._load_image(image_path)
        orig_h, orig_w = image.shape[:2]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        resized_h, resized_w = image.shape[1], image.shape[2]

        meta = {
            "patient_id": patient_id,
            "image_path": image_path,
            "dicom_path": dicom_path,
            "orig_height": orig_h,
            "orig_width": orig_w,
            "resized_height": resized_h,
            "resized_width": resized_w,
        }

        return image, meta


# ============================================================
# COLLATE
# ============================================================

def detection_collate_fn(batch):
    images = []
    targets = []
    metas = []

    for image, target, meta in batch:
        images.append(image)
        targets.append(target)
        metas.append(meta)

    images = torch.stack(images, dim=0)
    return images, targets, metas


def test_collate_fn(batch):
    images = []
    metas = []

    for image, meta in batch:
        images.append(image)
        metas.append(meta)

    images = torch.stack(images, dim=0)
    return images, metas


# ============================================================
# DATALOADER BUILDERS
# ============================================================

def create_train_val_dataloaders(
    train_csv: str,
    val_csv: str,
    cfg: RSNADetectionConfig
):
    train_ds = RSNADetectionTrainDataset(
        csv_path=train_csv,
        transform=get_train_transform(cfg),
        cfg=cfg
    )

    val_ds = RSNADetectionTrainDataset(
        csv_path=val_csv,
        transform=get_valid_transform(cfg),
        cfg=cfg
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        drop_last=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        drop_last=False
    )

    return train_ds, val_ds, train_loader, val_loader


def create_test_dataloader(
    test_csv: str,
    cfg: RSNADetectionConfig
):
    test_ds = RSNADetectionTestDataset(
        csv_path=test_csv,
        transform=get_test_transform(cfg),
        cfg=cfg
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=test_collate_fn,
        drop_last=False
    )

    return test_ds, test_loader


# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check_detection_loader(loader, num_batches: int = 1):
    print("=" * 80)
    print("[SANITY CHECK] Detection loader inspection started")
    print("=" * 80)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        images, targets, metas = batch

        print(f"[BATCH {batch_idx}] images.shape = {tuple(images.shape)}")

        for i in range(images.size(0)):
            img = images[i]
            tgt = targets[i]
            meta = metas[i]

            print("-" * 60)
            print(f"sample_index        : {i}")
            print(f"patient_id          : {meta['patient_id']}")
            print(f"image_path          : {meta['image_path']}")
            print(f"dicom_path          : {meta['dicom_path']}")
            print(f"image_shape         : {tuple(img.shape)}")
            print(f"num_boxes           : {len(tgt['boxes'])}")
            print(f"labels_shape        : {tuple(tgt['labels'].shape)}")
            print(f"boxes_shape         : {tuple(tgt['boxes'].shape)}")
            print(f"target_binary       : {meta.get('target_binary', 'NA')}")

            if len(tgt["boxes"]) > 0:
                print(f"first_box(cxcywh)   : {tgt['boxes'][0].tolist()}")

    print("=" * 80)
    print("[SANITY CHECK] Completed")
    print("=" * 80)