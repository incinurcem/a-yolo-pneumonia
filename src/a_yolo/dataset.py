"""
A-YOLO Dataset  –  RSNA Pneumonia Detection Challenge (Kaggle)

Expected CSV columns:
    patientId | x | y | width | height | Target
    (rows with Target=0 have NaN bbox; one patient may appear multiple times
     with different bbox rows)

Constructor signature (matches train.py):
    AYOLODataset(csv_path, img_dir, img_size, patch_size, is_train)
"""

import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AYOLODataset(Dataset):
    """
    RSNA-compatible Dataset for A-YOLO.

    Handles:
    - Multiple bbox rows per patient  → aggregated to the first valid box
    - Grayscale X-rays loaded as RGB  → compatible with ImageNet pretrained ViT
    - Anatomy-Aware Masking           → Gaussian-weighted center masking
    - CLAHE augmentation              → boosts low-contrast X-ray features
    """

    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, csv_path: str, img_dir: str,
                 img_size: int = 224, patch_size: int = 16,
                 is_train: bool = True, split_type: str = None):
        super().__init__()
        self.img_dir    = img_dir
        self.img_size   = img_size
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size
        self.is_train   = is_train

        # ── Load & normalise CSV ──────────────────────────────────────────────
        df = pd.read_csv(csv_path)

        if split_type and 'split' in df.columns:
            df = df[df['split'] == split_type].copy()
            print(f"📂 {split_type} split'i seçildi. Satır sayısı: {len(df)}")
        # Aggregate: one row per patient, first valid bbox wins
        self.records = self._build_records(df)

        # ── Albumentations pipeline ───────────────────────────────────────────
        bbox_params = A.BboxParams(
            format='coco',
            label_fields=['category_ids'],
            min_visibility=0.1,
        )
        if is_train:
            self.transform = A.Compose([
                # 🚀 DÜZELTME: height ve width yerine size kullanıyoruz
                A.RandomResizedCrop(
                    size=(img_size, img_size), # Artık tek parametre: size
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

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _build_records(df: pd.DataFrame) -> list[dict]:
        """
        Görseldeki sütun isimlerine (x_min, y_min, width, height) göre 
        kayıtları garantili şekilde oluşturur.
        """
        records = []
        # Sütun isimlerindeki olası boşlukları temizle
        df.columns = [c.strip() for c in df.columns]

        # Sütun isimlerini belirle (Senin görselindeki isimlerle eşliyoruz)
        target_col = 'target' if 'target' in df.columns else 'Target'
        
        # 🚀 KRİTİK DEĞİŞİKLİK: boxes_xywh yerine direkt sütunları kullanıyoruz
        x_col, y_col = 'x_min', 'y_min'
        w_col, h_col = 'width', 'height'
        
        for _, row in df.iterrows():
            pid = row['patientId']
            target = int(row[target_col])
            
            x, y, w, h = 0.0, 0.0, 0.0, 0.0
            
            # Eğer vakada zatürre varsa (target=1), koordinatları oku
            if target == 1:
                # pandas.notnull kontrolü ile NaN hatalarını engelle
                x = float(row[x_col]) if pd.notnull(row[x_col]) else 0.0
                y = float(row[y_col]) if pd.notnull(row[y_col]) else 0.0
                w = float(row[w_col]) if pd.notnull(row[w_col]) else 0.0
                h = float(row[h_col]) if pd.notnull(row[h_col]) else 0.0
            
            records.append({
                'patientId': pid,
                'target': target,
                'x': x, 'y': y,
                'w': w, 'h': h,
            })
            
       
        return records
    def _load_image(self, patient_id: str) -> np.ndarray:
        """Dosya yolu hatalarını ve senkronizasyon sorunlarını aşan zırhlı versiyon."""
        import os
        import cv2

        # 1. ID'yi temizle ve string olduğundan emin ol
        pid = str(patient_id).strip()
        
        # 2. Aranan muhtemel tam dosya adları
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

        # 🚀 3. Hala bulamadıysa (Drive senkronizasyon hatası ihtimali)
        # Bazen os.path.exists yanlış negatif verir, doğrudan listeyi kontrol etmeyi dene
        try:
            current_files = os.listdir(self.img_dir)
            for fname in possible_filenames:
                if fname in current_files:
                    # Dosya listede var ama exists 'False' diyorsa, yolu zorla oku
                    p = os.path.join(self.img_dir, fname)
                    img = cv2.imread(p)
                    if img is not None:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            pass

        raise FileNotFoundError(f"Kayıp Dosya: {pid} | Yol: {self.img_dir}")

    def _get_anatomy_mask(self) -> torch.Tensor:
        """
        Anatomy-Aware Masking:
        Gaussian distribution centred on the image (where lungs are),
        with small random jitter during training.
        Returns: LongTensor [num_masked]
        """
        y, x = np.ogrid[-1:1:complex(self.grid_size),
                        -1:1:complex(self.grid_size)]
        c_x = np.random.uniform(-0.1, 0.1) if self.is_train else 0.0
        c_y = np.random.uniform(-0.1, 0.1) if self.is_train else 0.0
        sigma = 0.5

        mask_probs = np.exp(-((x - c_x) ** 2 + (y - c_y) ** 2)
                            / (2 * sigma ** 2))
        mask_probs = mask_probs.flatten()
        mask_probs = mask_probs / mask_probs.sum()

        num_mask = int(0.75 * len(mask_probs))
        mask_indices = np.random.choice(
            len(mask_probs), size=num_mask, replace=False, p=mask_probs)
        return torch.tensor(mask_indices, dtype=torch.long)

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        patient_id = rec['patientId']

        # 1. Load image
        img = self._load_image(patient_id)
        orig_h, orig_w = img.shape[:2]

        # 2. Prepare bbox for augmentation pipeline
        bboxes, category_ids = [], []
        if rec['target'] == 1:
            x, y, w, h = rec['x'], rec['y'], rec['w'], rec['h']
            # Clip to image bounds
            x = max(0.0, x);  y = max(0.0, y)
            w = min(w, orig_w - x);  h = min(h, orig_h - y)
            if w > 1 and h > 1:
                bboxes.append([x, y, w, h])
                category_ids.append(1)

        # 3. Transform (image + bbox jointly)
        tfm = self.transform(image=img, bboxes=bboxes,
                             category_ids=category_ids)
        img_tensor = torch.from_numpy(tfm['image']).permute(2, 0, 1).float()

        # 4. Build detection targets
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

        # 5. Anatomy-Aware mask for SSL
        mask_idx = self._get_anatomy_mask()

        return {
            "image":        img_tensor,     # [3, H, W]
            "mask_indices": mask_idx,       # [num_masked]
            "label":        target_cls,     # scalar
            "bbox":         target_box,     # [4]  normalised [0,1]
            "patientId":    patient_id,
        }
