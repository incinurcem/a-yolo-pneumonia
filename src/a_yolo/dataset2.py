"""
A-YOLO Dataset — RSNA Pneumonia Detection Challenge (Kaggle)

Ablation desteği:
    mask_strategy = "gaussian"  → anatomi-aware (önerilen)
    mask_strategy = "random"    → standart MAE (He et al., 2022)
    mask_strategy = "none"      → train.py alpha=0 zorlayarak SSL'i kapatır
"""

import os
import ast
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AYOLODataset(Dataset):

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, csv_path: str, img_dir: str,
                 img_size: int = 224, patch_size: int = 16,
                 is_train: bool = True, split_type: str = None,
                 mask_strategy: str = "gaussian",
                 mask_ratio: float = 0.75):
        super().__init__()
        self.img_dir      = img_dir
        self.img_size     = img_size
        self.patch_size   = patch_size
        self.grid_size    = img_size // patch_size
        self.num_patches  = self.grid_size ** 2
        self.is_train     = is_train

        assert mask_strategy in ("gaussian", "random", "none"), \
            f"Unknown mask_strategy: {mask_strategy}"
        self.mask_strategy = mask_strategy
        self.mask_ratio    = mask_ratio
        self.num_mask      = int(self.num_patches * mask_ratio)

        df = pd.read_csv(csv_path)
        if split_type and 'split' in df.columns:
            df = df[df['split'] == split_type].copy()
            print(f"📂 {split_type} split'i seçildi. Satır sayısı: {len(df)}")
        self.records = self._build_records(df)

        bbox_params = A.BboxParams(
            format='coco',
            label_fields=['category_ids'],
            min_visibility=0.1,
        )
        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),
                A.CLAHE(clip_limit=2.0, p=0.4),
                A.Normalize(mean=self.MEAN, std=self.STD),
            ], bbox_params=bbox_params)
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.CLAHE(clip_limit=2.0, p=1.0),
                A.Normalize(mean=self.MEAN, std=self.STD),
            ], bbox_params=bbox_params)

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_records(df: pd.DataFrame) -> list:
        """RSNA master CSV: hem boxes_xywh hem x_min/y_min formatını destekler."""
        records = []
        df.columns = [c.strip() for c in df.columns]

        target_col     = 'target' if 'target' in df.columns else 'Target'
        has_boxes_xywh = 'boxes_xywh' in df.columns
        has_xmin       = all(c in df.columns for c in ['x_min', 'y_min'])
        parsed_count   = 0

        for _, row in df.iterrows():
            pid    = row['patientId']
            target = int(row[target_col])
            x, y, w, h = 0.0, 0.0, 0.0, 0.0
            parsed = False

            if target == 1:
                # Format 1: boxes_xywh
                if has_boxes_xywh and not parsed:
                    boxes_raw = row.get('boxes_xywh', '[]')
                    try:
                        if pd.notnull(boxes_raw):
                            boxes = ast.literal_eval(str(boxes_raw))
                            if boxes and len(boxes) > 0 and len(boxes[0]) >= 4:
                                x, y, w, h = (float(boxes[0][0]), float(boxes[0][1]),
                                              float(boxes[0][2]), float(boxes[0][3]))
                                parsed = True
                    except (ValueError, SyntaxError, TypeError):
                        pass

                # Format 2: x_min, y_min, width, height
                if has_xmin and not parsed:
                    try:
                        if pd.notnull(row['x_min']) and pd.notnull(row['y_min']):
                            x = float(row['x_min'])
                            y = float(row['y_min'])
                            w = float(row.get('width', 0))
                            h = float(row.get('height', 0))
                            if w > 0 and h > 0:
                                parsed = True
                    except (ValueError, TypeError):
                        pass

                if parsed:
                    parsed_count += 1

            records.append({
                'patientId': pid, 'target': target,
                'x': x, 'y': y, 'w': w, 'h': h,
            })

        print(f"✅ {len(records)} kayıt işlendi ({parsed_count} bbox parse edildi)")
        return records

    # ─────────────────────────────────────────────────────────────────────────
    def _load_image(self, patient_id: str) -> np.ndarray:
        pid = str(patient_id).strip()
        possible_filenames = [
            f"{pid}.png", f"{pid}.jpg", f"{pid}.jpeg",
            f"{pid}.PNG", f"{pid}.JPG"
        ]

        for fname in possible_filenames:
            p = os.path.join(self.img_dir, fname)
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            current_files = os.listdir(self.img_dir)
            for fname in possible_filenames:
                if fname in current_files:
                    p = os.path.join(self.img_dir, fname)
                    img = cv2.imread(p)
                    if img is not None:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        raise FileNotFoundError(f"Kayıp Dosya: {pid} | Yol: {self.img_dir}")

    # ── 🆕 ABLATION: 3 maskeleme stratejisi ─────────────────────────────────
    def _get_gaussian_mask(self) -> torch.Tensor:
        y, x = np.ogrid[-1:1:complex(self.grid_size),
                        -1:1:complex(self.grid_size)]
        c_x = np.random.uniform(-0.1, 0.1) if self.is_train else 0.0
        c_y = np.random.uniform(-0.1, 0.1) if self.is_train else 0.0
        sigma = 0.5

        mask_probs = np.exp(-((x - c_x) ** 2 + (y - c_y) ** 2) / (2 * sigma ** 2))
        mask_probs = mask_probs.flatten()
        mask_probs = mask_probs / mask_probs.sum()

        mask_indices = np.random.choice(
            self.num_patches, size=self.num_mask, replace=False, p=mask_probs)
        return torch.tensor(mask_indices, dtype=torch.long)

    def _get_random_mask(self) -> torch.Tensor:
        mask_indices = np.random.choice(
            self.num_patches, size=self.num_mask, replace=False)
        return torch.tensor(mask_indices, dtype=torch.long)

    def _get_mask_indices(self) -> torch.Tensor:
        if self.mask_strategy == "random":
            return self._get_random_mask()
        return self._get_gaussian_mask()

    def _get_anatomy_mask(self) -> torch.Tensor:
        return self._get_gaussian_mask()

    # ─────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        patient_id = rec['patientId']

        img = self._load_image(patient_id)
        orig_h, orig_w = img.shape[:2]

        bboxes, category_ids = [], []
        if rec['target'] == 1:
            x, y, w, h = rec['x'], rec['y'], rec['w'], rec['h']
            x = max(0.0, x); y = max(0.0, y)
            w = min(w, orig_w - x); h = min(h, orig_h - y)
            if w > 1 and h > 1:
                bboxes.append([x, y, w, h])
                category_ids.append(1)

        tfm = self.transform(image=img, bboxes=bboxes, category_ids=category_ids)
        img_tensor = torch.from_numpy(tfm['image']).permute(2, 0, 1).float()

        if len(tfm['bboxes']) > 0:
            rb = tfm['bboxes'][0]
            target_box = torch.tensor([
                rb[0] / self.img_size,
                rb[1] / self.img_size,
                rb[2] / self.img_size,
                rb[3] / self.img_size,
            ], dtype=torch.float32)
            target_cls = torch.tensor(1.0)
        else:
            target_box = torch.zeros(4, dtype=torch.float32)
            target_cls = torch.tensor(float(rec['target']))

        mask_idx = self._get_mask_indices()

        return {
            "image":        img_tensor,
            "mask_indices": mask_idx,
            "label":        target_cls,
            "bbox":         target_box,
            "patientId":    patient_id,
        }