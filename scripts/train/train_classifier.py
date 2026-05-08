# Wrapper entrypoint for classifier training
#s
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/train/train_classifier.py

RSNA Pneumonia / CXR tabanlı deep learning projesi için
binary classifier eğitim script'i.

Amaç:
- PNG görüntüler üzerinden pneumonia vs normal sınıflandırması yapmak
- Pretrained backbone ile fine-tune etmek
- Train / val / test split CSV'leri ile çalışmak
- Checkpoint, history, metrics, prediction CSV üretmek

Beklenen tipik dosya yapısı:
- data/images_png/<patient_id>.png
- data/splits/train.csv
- data/splits/val.csv
- data/splits/test.csv

CSV içinde desteklenen örnek kolonlar:
- image_path / path / png_path / filepath / file_path
veya
- patientId / patient_id / id / image_id

Label için desteklenen kolonlar:
- label / target / Target / class_id / pneumonia

Çalıştırma örneği:
python scripts/train/train_classifier.py \
    --base-config configs/base.yaml \
    --paths-config configs/paths.yaml \
    --task-config configs/classifier.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import yaml


# ============================================================
# PATH / ROOT
# ============================================================

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
else:
    PROJECT_ROOT = Path.cwd()


# ============================================================
# CONFIG HELPERS
# ============================================================

def read_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML dosyası sözlük değil: {path}")
    return data


def deep_merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_merged_config(
    base_config: Optional[Path],
    paths_config: Optional[Path],
    task_config: Optional[Path],
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for cfg_path in [base_config, paths_config, task_config]:
        cfg = deep_merge_dict(cfg, read_yaml(cfg_path))
    return cfg


def to_abs_path(path_like: Optional[str | Path]) -> Optional[Path]:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if len(rows) == 0:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# RANDOM / REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# DATAFRAME / SPLIT HELPERS
# ============================================================

def _find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def standardize_manifest(
    df: pd.DataFrame,
    image_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Çeşitli CSV formatlarını ortak biçime dönüştürür:
    image_path, label, patient_id, split
    """
    df = df.copy()

    patient_col = _find_first_existing_column(
        df, ["patientId", "patient_id", "id", "image_id"]
    )
    path_col = _find_first_existing_column(
        df, ["image_path", "path", "png_path", "filepath", "file_path"]
    )
    label_col = _find_first_existing_column(
        df, ["label", "target", "Target", "class_id", "pneumonia"]
    )
    split_col = _find_first_existing_column(
        df, ["split", "subset", "fold"]
    )
    class_name_col = _find_first_existing_column(
        df, ["class_name", "class", "diagnosis"]
    )

    # RSNA bbox CSV gibi tekrarlı hasta satırları varsa tekilleştir
    if patient_col is not None and df.duplicated(patient_col).any():
        agg_dict = {}
        if label_col is not None:
            agg_dict[label_col] = "max"
        if path_col is not None:
            agg_dict[path_col] = "first"
        if split_col is not None:
            agg_dict[split_col] = "first"
        if class_name_col is not None:
            agg_dict[class_name_col] = "first"
        if len(agg_dict) > 0:
            df = df.groupby(patient_col, as_index=False).agg(agg_dict)

    if path_col is None:
        if patient_col is None or image_dir is None:
            raise ValueError(
                "CSV içinde path yoksa patientId/patient_id ve image_dir verilmelidir."
            )
        df["image_path"] = df[patient_col].astype(str).apply(
            lambda x: str((image_dir / f"{x}.png").resolve())
        )
    else:
        resolved_paths = []
        for p in df[path_col].astype(str).tolist():
            p_obj = Path(p)
            if p_obj.is_absolute():
                resolved_paths.append(str(p_obj))
            else:
                if image_dir is not None:
                    resolved_paths.append(str((image_dir / p_obj).resolve()))
                else:
                    resolved_paths.append(str((PROJECT_ROOT / p_obj).resolve()))
        df["image_path"] = resolved_paths

    if label_col is not None:
        df["label"] = df[label_col].astype(float).astype(int)
    elif class_name_col is not None:
        mapping = {
            "normal": 0,
            "negative": 0,
            "no pneumonia": 0,
            "none": 0,
            "pneumonia": 1,
            "positive": 1,
            "opacity": 1,
        }
        df["label"] = (
            df[class_name_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )
    else:
        df["label"] = -1

    if patient_col is not None:
        df["patient_id"] = df[patient_col].astype(str)
    else:
        df["patient_id"] = df["image_path"].apply(lambda x: Path(x).stem)

    if split_col is not None:
        df["split"] = df[split_col].astype(str)
    else:
        df["split"] = ""

    df["exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    missing_count = int((~df["exists"]).sum())
    if missing_count > 0:
        print(f"[UYARI] {missing_count} adet görüntü bulunamadı, düşürülüyor.")
        df = df[df["exists"]].copy()

    df = df[["image_path", "label", "patient_id", "split"]].reset_index(drop=True)
    return df


def load_split_dataframe(
    split_name: str,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    split_cfg = data_cfg.get("splits", {})

    image_dir = to_abs_path(
        data_cfg.get("image_dir", paths_cfg.get("images_png_dir", "data/images_png"))
    )

    # Öncelik: ayrı split CSV
    split_csv = split_cfg.get(f"{split_name}_csv", None)
    if split_csv is None:
        split_csv = data_cfg.get(f"{split_name}_csv", None)

    if split_csv is not None:
        split_csv_path = to_abs_path(split_csv)
        if split_csv_path is None or not split_csv_path.exists():
            raise FileNotFoundError(f"{split_name} CSV bulunamadı: {split_csv}")
        df = pd.read_csv(split_csv_path)
        return standardize_manifest(df, image_dir=image_dir)

    # Alternatif: tek manifest + split kolonu
    manifest_csv = data_cfg.get("manifest_csv", None)
    if manifest_csv is None:
        manifest_csv = paths_cfg.get("manifest_csv", None)

    if manifest_csv is not None:
        manifest_path = to_abs_path(manifest_csv)
        if manifest_path is None or not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV bulunamadı: {manifest_csv}")
        manifest_df = pd.read_csv(manifest_path)
        manifest_df = standardize_manifest(manifest_df, image_dir=image_dir)
        if "split" not in manifest_df.columns or manifest_df["split"].nunique() <= 1:
            raise ValueError(
                "Tek manifest kullanılıyor ama split/subset kolonu bulunamadı."
            )
        subset_df = manifest_df[
            manifest_df["split"].astype(str).str.lower() == split_name.lower()
        ].copy()
        if len(subset_df) == 0:
            raise ValueError(f"Manifest içinde '{split_name}' split'i boş.")
        return subset_df.reset_index(drop=True)

    # Son fallback: varsayılan split yolu
    default_csv = PROJECT_ROOT / "data" / "splits" / f"{split_name}.csv"
    if default_csv.exists():
        df = pd.read_csv(default_csv)
        return standardize_manifest(df, image_dir=image_dir)

    raise FileNotFoundError(
        f"{split_name} split'i için uygun CSV bulunamadı. "
        "configs/classifier.yaml veya configs/paths.yaml içinde split yollarını tanımlayın."
    )


# ============================================================
# DATASET
# ============================================================

class CXRClassificationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int = 512,
        in_channels: int = 3,
        train: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.train = bool(train)

        if mean is None:
            mean = [0.485, 0.456, 0.406] if self.in_channels == 3 else [0.5]
        if std is None:
            std = [0.229, 0.224, 0.225] if self.in_channels == 3 else [0.25]

        self.transform = self._build_transform(mean, std)

    def _build_transform(self, mean: List[float], std: List[float]) -> transforms.Compose:
        tfms: List[Any] = [
            transforms.Resize((self.image_size, self.image_size)),
        ]

        if self.train:
            tfms.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=7),
                ]
            )

        tfms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        return transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        image_path = str(row["image_path"])
        label = int(row["label"])
        patient_id = str(row["patient_id"])

        image = Image.open(image_path)

        if self.in_channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "patient_id": patient_id,
            "image_path": image_path,
        }


# ============================================================
# MODEL HELPERS
# ============================================================

def _set_nested_module(root_module: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root_module
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module  # type: ignore[index]
    else:
        setattr(parent, last, new_module)


def replace_first_conv(model: nn.Module, in_channels: int) -> nn.Module:
    if in_channels == 3:
        return model

    first_conv_name = None
    first_conv_module = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv_name = name
            first_conv_module = module
            break

    if first_conv_name is None or first_conv_module is None:
        raise RuntimeError("Model içinde Conv2d bulunamadı, ilk katman değiştirilemedi.")

    old_conv = first_conv_module
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        if old_conv.weight.shape[1] == 3 and in_channels == 1:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        elif old_conv.weight.shape[1] == 3 and in_channels > 3:
            repeated = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            new_conv.weight.copy_(repeated / max(in_channels, 1))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    _set_nested_module(model, first_conv_name, new_conv)
    return model


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)
        feats = self.dropout(feats)
        logits = self.head(feats).squeeze(1)
        return logits


def build_backbone(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    model_name = model_name.lower()

    if model_name == "resnet18":
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    if model_name == "resnet34":
        try:
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        except Exception:
            backbone = models.resnet34(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    if model_name == "resnet50":
        try:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        except Exception:
            backbone = models.resnet50(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    if model_name == "densenet121":
        try:
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        except Exception:
            backbone = models.densenet121(pretrained=pretrained)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        return backbone, in_features

    if model_name == "efficientnet_b0":
        try:
            backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
        except Exception:
            backbone = models.efficientnet_b0(pretrained=pretrained)
        if isinstance(backbone.classifier, nn.Sequential):
            in_features = backbone.classifier[-1].in_features
        else:
            raise RuntimeError("efficientnet_b0 classifier yapısı beklenen formatta değil.")
        backbone.classifier = nn.Identity()
        return backbone, in_features

    if model_name == "convnext_tiny":
        try:
            backbone = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            )
        except Exception:
            backbone = models.convnext_tiny(pretrained=pretrained)
        if isinstance(backbone.classifier, nn.Sequential):
            in_features = backbone.classifier[-1].in_features
        else:
            raise RuntimeError("convnext_tiny classifier yapısı beklenen formatta değil.")
        backbone.classifier = nn.Identity()
        return backbone, in_features

    raise ValueError(
        f"Desteklenmeyen model_name: {model_name}. "
        "Desteklenenler: resnet18, resnet34, resnet50, densenet121, efficientnet_b0, convnext_tiny"
    )


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "resnet50"))
    pretrained = bool(model_cfg.get("pretrained", True))
    in_channels = int(model_cfg.get("in_channels", 3))
    dropout = float(model_cfg.get("dropout", 0.2))

    backbone, in_features = build_backbone(model_name=model_name, pretrained=pretrained)
    backbone = replace_first_conv(backbone, in_channels=in_channels)

    model = BinaryClassifier(
        backbone=backbone,
        in_features=in_features,
        dropout=dropout,
    )
    return model


def set_backbone_requires_grad(model: BinaryClassifier, requires_grad: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


# ============================================================
# METRICS
# ============================================================

def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    metrics["threshold"] = float(threshold)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["tp"] = float(tp)
        metrics["tn"] = float(tn)
        metrics["fp"] = float(fp)
        metrics["fn"] = float(fn)
    else:
        metrics["specificity"] = 0.0
        metrics["sensitivity"] = 0.0
        metrics["tp"] = 0.0
        metrics["tn"] = 0.0
        metrics["fp"] = 0.0
        metrics["fn"] = 0.0

    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    best_threshold = float(thresholds[best_idx])

    if math.isinf(best_threshold) or math.isnan(best_threshold):
        return 0.5

    return max(0.0, min(1.0, best_threshold))


# ============================================================
# TRAIN / EVAL LOOP
# ============================================================

def make_device(requested_device: Optional[str] = None) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
) -> Tuple[float, np.ndarray, np.ndarray, List[str], List[str]]:
    model.train()

    running_loss = 0.0
    all_labels: List[float] = []
    all_probs: List[float] = []
    all_ids: List[str] = []
    all_paths: List[str] = []

    autocast_enabled = amp_enabled and device.type == "cuda"

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        images = batch["image"]
        labels = batch["label"].float()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits).detach().cpu().numpy()

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.tolist())
        all_ids.extend(batch["patient_id"])
        all_paths.extend(batch["image_path"])

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    return (
        float(epoch_loss),
        np.asarray(all_labels),
        np.asarray(all_probs),
        all_ids,
        all_paths,
    )


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, np.ndarray, np.ndarray, List[str], List[str]]:
    model.eval()

    running_loss = 0.0
    all_labels: List[float] = []
    all_probs: List[float] = []
    all_ids: List[str] = []
    all_paths: List[str] = []

    autocast_enabled = amp_enabled and device.type == "cuda"

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        images = batch["image"]
        labels = batch["label"].float()

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())
        all_ids.extend(batch["patient_id"])
        all_paths.extend(batch["image_path"])

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    return (
        float(epoch_loss),
        np.asarray(all_labels),
        np.asarray(all_probs),
        all_ids,
        all_paths,
    )


def build_optimizer(
    model: nn.Module,
    cfg: Dict[str, Any],
) -> torch.optim.Optimizer:
    optim_cfg = cfg.get("optimizer", {})
    lr = float(optim_cfg.get("lr", 1e-4))
    weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
    name = str(optim_cfg.get("name", "adamw")).lower()

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(optim_cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Desteklenmeyen optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
):
    sched_cfg = cfg.get("scheduler", {})
    name = str(sched_cfg.get("name", "cosine")).lower()

    if name in ["none", "null", "false", "off"]:
        return None

    if name == "cosine":
        epochs = int(cfg.get("training", {}).get("epochs", 20))
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )

    if name == "plateau":
        factor = float(sched_cfg.get("factor", 0.5))
        patience = int(sched_cfg.get("patience", 2))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        )

    if name == "step":
        step_size = int(sched_cfg.get("step_size", 5))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    raise ValueError(f"Desteklenmeyen scheduler: {name}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_score: float,
    best_threshold: float,
    cfg: Dict[str, Any],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict(),
            "best_score": best_score,
            "best_threshold": best_threshold,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary classifier training script")

    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Base config YAML yolu",
    )
    parser.add_argument(
        "--paths-config",
        type=str,
        default="configs/paths.yaml",
        help="Paths config YAML yolu",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="configs/classifier.yaml",
        help="Classifier config YAML yolu",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Kaldığın checkpoint'ten devam et",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu / mps gibi device override",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Çalışma adı override",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_merged_config(
        base_config=to_abs_path(args.base_config),
        paths_config=to_abs_path(args.paths_config),
        task_config=to_abs_path(args.task_config),
    )

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})

    device = make_device(args.device)
    print(f"[INFO] Device: {device}")

    run_name = args.run_name or output_cfg.get("run_name") or f"classifier_{int(time.time())}"

    output_dir = to_abs_path(
        output_cfg.get("classifier_dir", f"outputs/classifier/{run_name}")
    )
    if output_dir is None:
        raise RuntimeError("output_dir çözümlenemedi.")
    ensure_dir(output_dir)

    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    pred_dir = ensure_dir(output_dir / "predictions")
    metrics_dir = ensure_dir(output_dir / "metrics")
    history_dir = ensure_dir(output_dir / "history")

    save_json(cfg, output_dir / "merged_config.json")

    epochs = int(training_cfg.get("epochs", 20))
    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 4))
    image_size = int(training_cfg.get("image_size", model_cfg.get("image_size", 512)))
    amp_enabled = bool(training_cfg.get("amp", True))
    persistent_workers = bool(training_cfg.get("persistent_workers", num_workers > 0))
    pin_memory = bool(training_cfg.get("pin_memory", device.type == "cuda"))
    freeze_backbone_epochs = int(training_cfg.get("freeze_backbone_epochs", 0))
    optimize_threshold = bool(training_cfg.get("optimize_threshold_on_val", True))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 10))

    in_channels = int(model_cfg.get("in_channels", 3))
    mean = model_cfg.get("mean", None)
    std = model_cfg.get("std", None)

    train_df = load_split_dataframe("train", cfg)
    val_df = load_split_dataframe("val", cfg)
    test_df = load_split_dataframe("test", cfg)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Train/val/test splitlerinden biri boş.")

    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = CXRClassificationDataset(
        dataframe=train_df,
        image_size=image_size,
        in_channels=in_channels,
        train=True,
        mean=mean,
        std=std,
    )
    val_ds = CXRClassificationDataset(
        dataframe=val_df,
        image_size=image_size,
        in_channels=in_channels,
        train=False,
        mean=mean,
        std=std,
    )
    test_ds = CXRClassificationDataset(
        dataframe=test_df,
        image_size=image_size,
        in_channels=in_channels,
        train=False,
        mean=mean,
        std=std,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=seed_worker,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    model = build_model(cfg).to(device)

    if freeze_backbone_epochs > 0:
        set_backbone_requires_grad(model, False)
        print(f"[INFO] Backbone ilk {freeze_backbone_epochs} epoch freeze edildi.")

    # class imbalance için pos_weight
    loss_cfg = cfg.get("loss", {})
    auto_pos_weight = bool(loss_cfg.get("auto_pos_weight", True))

    pos_weight_tensor = None
    if auto_pos_weight:
        labels_np = train_df["label"].astype(int).to_numpy()
        pos_count = float((labels_np == 1).sum())
        neg_count = float((labels_np == 0).sum())
        if pos_count > 0:
            pos_weight_value = neg_count / pos_count
            pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
            print(f"[INFO] pos_weight = {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor if pos_weight_tensor is not None else None
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))

    start_epoch = 0
    best_score = -float("inf")
    best_threshold = 0.5
    epochs_without_improvement = 0

    if args.resume is not None:
        ckpt = load_checkpoint(
            checkpoint_path=to_abs_path(args.resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device.type,
        )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_score = float(ckpt.get("best_score", -float("inf")))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        print(f"[INFO] Resume edildi. Start epoch = {start_epoch}")

    history_rows: List[Dict[str, Any]] = []

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
            set_backbone_requires_grad(model, True)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg)
            print(f"[INFO] Epoch {epoch}: backbone unfreeze edildi ve optimizer resetlendi.")

        train_loss, train_labels, train_probs, _, _ = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        train_metrics = compute_binary_metrics(train_labels, train_probs, threshold=0.5)

        val_loss, val_labels, val_probs, val_ids, val_paths = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        epoch_threshold = 0.5
        if optimize_threshold:
            epoch_threshold = find_best_threshold(val_labels, val_probs)

        val_metrics = compute_binary_metrics(val_labels, val_probs, threshold=epoch_threshold)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # best score seçim mantığı
        current_score = val_metrics.get("roc_auc", float("nan"))
        if math.isnan(current_score):
            current_score = -val_loss

        is_best = current_score > best_score
        if is_best:
            best_score = float(current_score)
            best_threshold = float(epoch_threshold)
            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_threshold=best_threshold,
                cfg=cfg,
            )

            best_val_rows = []
            val_preds = (val_probs >= best_threshold).astype(int)
            for pid, path, label, prob, pred in zip(
                val_ids, val_paths, val_labels.tolist(), val_probs.tolist(), val_preds.tolist()
            ):
                best_val_rows.append(
                    {
                        "patient_id": pid,
                        "image_path": path,
                        "label": int(label),
                        "prob": float(prob),
                        "pred": int(pred),
                    }
                )
            save_csv(best_val_rows, pred_dir / "best_val_predictions.csv")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_score=best_score,
            best_threshold=best_threshold,
            cfg=cfg,
        )

        epoch_time = time.time() - epoch_start_time

        row = {
            "epoch": epoch,
            "lr": float(current_lr),
            "time_sec": float(epoch_time),
            "train_loss": float(train_loss),
            "train_acc": float(train_metrics["accuracy"]),
            "train_precision": float(train_metrics["precision"]),
            "train_recall": float(train_metrics["recall"]),
            "train_f1": float(train_metrics["f1"]),
            "train_roc_auc": float(train_metrics["roc_auc"]) if not math.isnan(train_metrics["roc_auc"]) else np.nan,
            "train_pr_auc": float(train_metrics["pr_auc"]) if not math.isnan(train_metrics["pr_auc"]) else np.nan,
            "val_loss": float(val_loss),
            "val_acc": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_specificity": float(val_metrics["specificity"]),
            "val_sensitivity": float(val_metrics["sensitivity"]),
            "val_roc_auc": float(val_metrics["roc_auc"]) if not math.isnan(val_metrics["roc_auc"]) else np.nan,
            "val_pr_auc": float(val_metrics["pr_auc"]) if not math.isnan(val_metrics["pr_auc"]) else np.nan,
            "val_threshold": float(epoch_threshold),
            "is_best": int(is_best),
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_dir / "history.csv", index=False)

        print(
            f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['roc_auc'] if not math.isnan(val_metrics['roc_auc']) else 'nan'} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"thr={epoch_threshold:.4f} "
            f"lr={current_lr:.8f}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"[INFO] Early stopping tetiklendi. patience={early_stopping_patience}")
            break

    # En iyi modeli yükle
    best_ckpt_path = checkpoint_dir / "best.pt"
    if not best_ckpt_path.exists():
        raise FileNotFoundError("best.pt oluşmadı. Eğitim başarısız olmuş olabilir.")

    best_ckpt = load_checkpoint(
        checkpoint_path=best_ckpt_path,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        map_location=device.type,
    )
    best_threshold = float(best_ckpt.get("best_threshold", best_threshold))

    # Final val değerlendirmesi
    val_loss, val_labels, val_probs, val_ids, val_paths = evaluate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )

    if optimize_threshold:
        best_threshold = find_best_threshold(val_labels, val_probs)

    final_val_metrics = compute_binary_metrics(val_labels, val_probs, threshold=best_threshold)

    # Test değerlendirmesi
    test_loss, test_labels, test_probs, test_ids, test_paths = evaluate_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )
    final_test_metrics = compute_binary_metrics(test_labels, test_probs, threshold=best_threshold)
    final_test_metrics["loss"] = float(test_loss)

    # prediction CSV
    test_preds = (test_probs >= best_threshold).astype(int)
    test_rows = []
    for pid, path, label, prob, pred in zip(
        test_ids, test_paths, test_labels.tolist(), test_probs.tolist(), test_preds.tolist()
    ):
        test_rows.append(
            {
                "patient_id": pid,
                "image_path": path,
                "label": int(label),
                "prob": float(prob),
                "pred": int(pred),
            }
        )
    save_csv(test_rows, pred_dir / "test_predictions.csv")

    final_summary = {
        "run_name": run_name,
        "best_threshold": float(best_threshold),
        "best_score": float(best_score),
        "best_checkpoint": str(best_ckpt_path),
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "device": str(device),
    }

    save_json(final_summary, metrics_dir / "summary.json")
    pd.DataFrame([final_val_metrics]).to_csv(metrics_dir / "val_metrics.csv", index=False)
    pd.DataFrame([final_test_metrics]).to_csv(metrics_dir / "test_metrics.csv", index=False)

    print("\n[INFO] Eğitim tamamlandı.")
    print(f"[INFO] En iyi checkpoint : {best_ckpt_path}")
    print(f"[INFO] Test metrics      : {metrics_dir / 'test_metrics.csv'}")
    print(f"[INFO] Test predictions  : {pred_dir / 'test_predictions.csv'}")
    print(f"[INFO] Summary           : {metrics_dir / 'summary.json'}")


if __name__ == "__main__":
    main()