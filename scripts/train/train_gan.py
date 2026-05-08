# Wrapper entrypoint for GAN training
#s

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/train/train_gan.py

RSNA / CXR deep learning projesi için GAN tabanlı reconstruction modeli.
Amaç:
- Normal akciğer görüntüleri üzerinde reconstruction öğrenmek
- Reconstruction error üzerinden anomaly localization / anomaly score üretmek
- Image-level anomaly AUROC hesaplamak
- Checkpoint, history, recon örnekleri ve skor CSV üretmek

Temel fikir:
- Generator: U-Net benzeri autoencoder
- Discriminator: PatchGAN benzeri discriminator
- Eğitim: adversarial + L1 reconstruction + edge consistency loss
- Kullanım: pneumonia görüntülerinde reconstruction bozulduğu için fark haritası anomaliyi vurgular

Beklenen tipik dosya yapısı:
- data/images_png/<patient_id>.png
- data/splits/train.csv
- data/splits/val.csv
- data/splits/test.csv

Not:
- Train setten sadece normal örnekler ile GAN eğitilir.
- Val/test set ise anomaly scoring için tüm sınıfları içerebilir.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
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
# DATA HELPERS
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
    df = df.copy()

    patient_col = _find_first_existing_column(df, ["patientId", "patient_id", "id", "image_id"])
    path_col = _find_first_existing_column(df, ["image_path", "path", "png_path", "filepath", "file_path"])
    label_col = _find_first_existing_column(df, ["label", "target", "Target", "class_id", "pneumonia"])
    split_col = _find_first_existing_column(df, ["split", "subset", "fold"])
    class_name_col = _find_first_existing_column(df, ["class_name", "class", "diagnosis"])

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
            raise ValueError("Path kolonu yoksa patient_id ve image_dir olmalı.")
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
            .str.lower()
            .str.strip()
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
    missing = int((~df["exists"]).sum())
    if missing > 0:
        print(f"[UYARI] {missing} görüntü bulunamadı, düşürülüyor.")
        df = df[df["exists"]].copy()

    df = df[["image_path", "label", "patient_id", "split"]].reset_index(drop=True)
    return df


def load_split_dataframe(split_name: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    split_cfg = data_cfg.get("splits", {})

    image_dir = to_abs_path(
        data_cfg.get("image_dir", paths_cfg.get("images_png_dir", "data/images_png"))
    )

    split_csv = split_cfg.get(f"{split_name}_csv", None)
    if split_csv is None:
        split_csv = data_cfg.get(f"{split_name}_csv", None)

    if split_csv is not None:
        split_path = to_abs_path(split_csv)
        if split_path is None or not split_path.exists():
            raise FileNotFoundError(f"{split_name} CSV bulunamadı: {split_csv}")
        df = pd.read_csv(split_path)
        return standardize_manifest(df, image_dir=image_dir)

    manifest_csv = data_cfg.get("manifest_csv", None)
    if manifest_csv is None:
        manifest_csv = paths_cfg.get("manifest_csv", None)

    if manifest_csv is not None:
        manifest_path = to_abs_path(manifest_csv)
        if manifest_path is None or not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV bulunamadı: {manifest_csv}")
        df = pd.read_csv(manifest_path)
        df = standardize_manifest(df, image_dir=image_dir)
        subset_df = df[df["split"].astype(str).str.lower() == split_name.lower()].copy()
        if len(subset_df) == 0:
            raise ValueError(f"Manifest içinde '{split_name}' split'i boş.")
        return subset_df.reset_index(drop=True)

    default_csv = PROJECT_ROOT / "data" / "splits" / f"{split_name}.csv"
    if default_csv.exists():
        df = pd.read_csv(default_csv)
        return standardize_manifest(df, image_dir=image_dir)

    raise FileNotFoundError(
        f"{split_name} split CSV bulunamadı. configs içinden yol verin."
    )


# ============================================================
# DATASET
# ============================================================

class CXRGANDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int = 256,
        in_channels: int = 1,
        train: bool = False,
    ) -> None:
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.train = bool(train)

        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        tfms: List[Any] = [transforms.Resize((self.image_size, self.image_size))]

        if self.train:
            tfms.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=5),
                ]
            )

        tfms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5] * self.in_channels,
                    std=[0.5] * self.in_channels,
                ),
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

        img = Image.open(image_path)
        if self.in_channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.float32),
            "patient_id": patient_id,
            "image_path": image_path,
        }


# ============================================================
# MODELS
# ============================================================

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: bool = True,
        act: str = "leaky",
        dropout: float = 0.0,
        stride: int = 1,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not norm)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            raise ValueError(f"Desteklenmeyen activation: {act}")

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not norm),
            ]
        )
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if act == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(
            in_channels, out_channels, norm=True, act="leaky", dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            norm=True,
            act="relu",
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_block = ConvBlock(in_channels, base_channels, norm=True, act="relu", dropout=0.0)
        self.down1 = DownBlock(base_channels, base_channels * 2, dropout=0.0)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, dropout=0.0)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        self.down4 = DownBlock(base_channels * 8, base_channels * 8, dropout=dropout)

        self.up1 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4, dropout=dropout)
        self.up2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, dropout=dropout)
        self.up3 = UpBlock(base_channels * 2, base_channels * 2, base_channels, dropout=0.0)
        self.up4 = UpBlock(base_channels, base_channels, base_channels, dropout=0.0)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)

        out = self.out_conv(y)
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================
# LOSSES / SCORES
# ============================================================

class SobelEdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def _grad(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1)
        weight_x = self.kernel_x.repeat(c, 1, 1, 1)
        weight_y = self.kernel_y.repeat(c, 1, 1, 1)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=c)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=c)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_grad = self._grad(pred)
        target_grad = self._grad(target)
        return F.l1_loss(pred_grad, target_grad)


def anomaly_map_and_score(x: torch.Tensor, recon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x ve recon [-1, 1] aralığında.
    Çıktı:
    - anomaly_map: [B,1,H,W]
    - score: [B]
    """
    diff = torch.abs(x - recon)
    if diff.size(1) > 1:
        diff_map = diff.mean(dim=1, keepdim=True)
    else:
        diff_map = diff
    score = diff_map.flatten(1).mean(dim=1)
    return diff_map, score


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


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


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0


def save_reconstruction_panel(
    x: torch.Tensor,
    recon: torch.Tensor,
    anomaly_map: torch.Tensor,
    save_path: str | Path,
    max_items: int = 8,
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    x = denorm_to_01(x.detach().cpu())
    recon = denorm_to_01(recon.detach().cpu())
    anomaly_map = anomaly_map.detach().cpu()

    anomaly_map = anomaly_map - anomaly_map.min()
    anomaly_map = anomaly_map / (anomaly_map.max() + 1e-8)

    n = min(max_items, x.size(0))
    panel = torch.cat(
        [
            x[:n],
            recon[:n],
            anomaly_map[:n],
        ],
        dim=0,
    )
    save_image(panel, save_path, nrow=n)


def build_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    cfg: Dict[str, Any],
) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    optim_cfg = cfg.get("optimizer", {})
    g_lr = float(optim_cfg.get("g_lr", optim_cfg.get("lr", 2e-4)))
    d_lr = float(optim_cfg.get("d_lr", optim_cfg.get("lr", 2e-4)))
    beta1 = float(optim_cfg.get("beta1", 0.5))
    beta2 = float(optim_cfg.get("beta2", 0.999))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    g_optim = torch.optim.Adam(
        generator.parameters(),
        lr=g_lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    d_optim = torch.optim.Adam(
        discriminator.parameters(),
        lr=d_lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    return g_optim, d_optim


def save_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    g_scaler: torch.cuda.amp.GradScaler,
    d_scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_metric: float,
    cfg: Dict[str, Any],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "g_scaler_state_dict": g_scaler.state_dict(),
            "d_scaler_state_dict": d_scaler.state_dict(),
            "best_metric": best_metric,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    generator: nn.Module,
    discriminator: Optional[nn.Module] = None,
    g_optimizer: Optional[torch.optim.Optimizer] = None,
    d_optimizer: Optional[torch.optim.Optimizer] = None,
    g_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    d_scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    generator.load_state_dict(ckpt["generator_state_dict"])

    if discriminator is not None and ckpt.get("discriminator_state_dict") is not None:
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])

    if g_optimizer is not None and ckpt.get("g_optimizer_state_dict") is not None:
        g_optimizer.load_state_dict(ckpt["g_optimizer_state_dict"])

    if d_optimizer is not None and ckpt.get("d_optimizer_state_dict") is not None:
        d_optimizer.load_state_dict(ckpt["d_optimizer_state_dict"])

    if g_scaler is not None and ckpt.get("g_scaler_state_dict") is not None:
        g_scaler.load_state_dict(ckpt["g_scaler_state_dict"])

    if d_scaler is not None and ckpt.get("d_scaler_state_dict") is not None:
        d_scaler.load_state_dict(ckpt["d_scaler_state_dict"])

    return ckpt


def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    adv_criterion: nn.Module,
    recon_criterion: nn.Module,
    edge_criterion: nn.Module,
    device: torch.device,
    g_scaler: torch.cuda.amp.GradScaler,
    d_scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    lambda_adv: float,
    lambda_l1: float,
    lambda_edge: float,
) -> Dict[str, float]:
    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_recon_loss = 0.0
    total_edge_loss = 0.0
    total_adv_loss = 0.0

    autocast_enabled = amp_enabled and device.type == "cuda"

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        real = batch["image"]

        # ----------------------------------
        # 1) Discriminator update
        # ----------------------------------
        d_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            fake = generator(real).detach()

            d_real = discriminator(real)
            d_fake = discriminator(fake)

            real_targets = torch.ones_like(d_real)
            fake_targets = torch.zeros_like(d_fake)

            d_loss_real = adv_criterion(d_real, real_targets)
            d_loss_fake = adv_criterion(d_fake, fake_targets)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        d_scaler.scale(d_loss).backward()
        d_scaler.step(d_optimizer)
        d_scaler.update()

        # ----------------------------------
        # 2) Generator update
        # ----------------------------------
        g_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            recon = generator(real)
            d_fake_for_g = discriminator(recon)

            adv_targets = torch.ones_like(d_fake_for_g)
            adv_loss = adv_criterion(d_fake_for_g, adv_targets)
            recon_loss = recon_criterion(recon, real)
            edge_loss = edge_criterion(recon, real)

            g_loss = (
                lambda_adv * adv_loss +
                lambda_l1 * recon_loss +
                lambda_edge * edge_loss
            )

        g_scaler.scale(g_loss).backward()
        g_scaler.step(g_optimizer)
        g_scaler.update()

        batch_size = real.size(0)
        total_g_loss += g_loss.item() * batch_size
        total_d_loss += d_loss.item() * batch_size
        total_recon_loss += recon_loss.item() * batch_size
        total_edge_loss += edge_loss.item() * batch_size
        total_adv_loss += adv_loss.item() * batch_size

    denom = max(len(loader.dataset), 1)
    return {
        "g_loss": float(total_g_loss / denom),
        "d_loss": float(total_d_loss / denom),
        "recon_l1": float(total_recon_loss / denom),
        "edge_loss": float(total_edge_loss / denom),
        "adv_loss": float(total_adv_loss / denom),
    }


@torch.no_grad()
def evaluate_generator(
    generator: nn.Module,
    loader: DataLoader,
    recon_criterion: nn.Module,
    edge_criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, Any]:
    generator.eval()

    total_recon_loss = 0.0
    total_edge_loss = 0.0

    all_labels: List[int] = []
    all_scores: List[float] = []
    all_ids: List[str] = []
    all_paths: List[str] = []

    last_batch_x = None
    last_batch_recon = None
    last_batch_map = None

    autocast_enabled = amp_enabled and device.type == "cuda"

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        x = batch["image"]
        labels = batch["label"].long()

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            recon = generator(x)
            recon_loss = recon_criterion(recon, x)
            edge_loss = edge_criterion(recon, x)

        amap, score = anomaly_map_and_score(x, recon)

        total_recon_loss += recon_loss.item() * x.size(0)
        total_edge_loss += edge_loss.item() * x.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_scores.extend(score.cpu().numpy().tolist())
        all_ids.extend(batch["patient_id"])
        all_paths.extend(batch["image_path"])

        last_batch_x = x
        last_batch_recon = recon
        last_batch_map = amap

    denom = max(len(loader.dataset), 1)
    recon_l1 = float(total_recon_loss / denom)
    edge_l1 = float(total_edge_loss / denom)

    y_true = np.asarray(all_labels).astype(int)
    y_score = np.asarray(all_scores).astype(float)

    image_auroc = safe_auc(y_true, y_score)
    image_ap = safe_ap(y_true, y_score)

    return {
        "recon_l1": recon_l1,
        "edge_l1": edge_l1,
        "image_auroc": image_auroc,
        "image_ap": image_ap,
        "labels": y_true,
        "scores": y_score,
        "patient_ids": all_ids,
        "image_paths": all_paths,
        "sample_x": last_batch_x,
        "sample_recon": last_batch_recon,
        "sample_map": last_batch_map,
    }


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAN training for anomaly localization")

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
        help="GAN ayarları classifier.yaml içinde de tutulabilir",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu / mps override",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name override",
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

    gan_cfg = cfg.get("gan", {})
    if len(gan_cfg) == 0:
        gan_cfg = cfg.get("model", {})

    training_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    device = make_device(args.device)
    print(f"[INFO] Device: {device}")

    run_name = args.run_name or output_cfg.get("gan_run_name") or f"gan_{int(time.time())}"

    output_dir = to_abs_path(
        output_cfg.get("gan_dir", f"outputs/gan/{run_name}")
    )
    if output_dir is None:
        raise RuntimeError("GAN output_dir çözümlenemedi.")
    ensure_dir(output_dir)

    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    history_dir = ensure_dir(output_dir / "history")
    recon_dir = ensure_dir(output_dir / "reconstructions")
    metrics_dir = ensure_dir(output_dir / "metrics")
    scores_dir = ensure_dir(output_dir / "scores")

    save_json(cfg, output_dir / "merged_config.json")

    epochs = int(training_cfg.get("gan_epochs", training_cfg.get("epochs", 30)))
    batch_size = int(training_cfg.get("gan_batch_size", training_cfg.get("batch_size", 16)))
    num_workers = int(training_cfg.get("num_workers", 4))
    image_size = int(gan_cfg.get("image_size", training_cfg.get("image_size", 256)))
    in_channels = int(gan_cfg.get("in_channels", cfg.get("model", {}).get("in_channels", 1)))
    base_channels = int(gan_cfg.get("base_channels", 64))
    dropout = float(gan_cfg.get("dropout", 0.1))
    amp_enabled = bool(training_cfg.get("amp", True))
    persistent_workers = bool(training_cfg.get("persistent_workers", num_workers > 0))
    pin_memory = bool(training_cfg.get("pin_memory", device.type == "cuda"))
    early_stopping_patience = int(training_cfg.get("gan_early_stopping_patience", 12))

    lambda_adv = float(gan_cfg.get("lambda_adv", 0.01))
    lambda_l1 = float(gan_cfg.get("lambda_l1", 1.0))
    lambda_edge = float(gan_cfg.get("lambda_edge", 0.2))

    sample_every = int(gan_cfg.get("sample_every", 1))
    drop_last = bool(gan_cfg.get("drop_last", True))

    train_df = load_split_dataframe("train", cfg)
    val_df = load_split_dataframe("val", cfg)
    test_df = load_split_dataframe("test", cfg)

    # GAN train set sadece normal örneklerden oluşmalı
    normal_train_df = train_df[train_df["label"].astype(int) == 0].copy()
    if len(normal_train_df) == 0:
        raise RuntimeError("GAN eğitimi için normal train örneği bulunamadı.")

    print(
        f"[INFO] GAN Train(normal only): {len(normal_train_df)} | "
        f"Val(all): {len(val_df)} | Test(all): {len(test_df)}"
    )

    train_ds = CXRGANDataset(
        dataframe=normal_train_df,
        image_size=image_size,
        in_channels=in_channels,
        train=True,
    )
    val_ds = CXRGANDataset(
        dataframe=val_df,
        image_size=image_size,
        in_channels=in_channels,
        train=False,
    )
    test_ds = CXRGANDataset(
        dataframe=test_df,
        image_size=image_size,
        in_channels=in_channels,
        train=False,
    )

    generator_seed = torch.Generator()
    generator_seed.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=generator_seed,
        drop_last=drop_last,
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

    generator = UNetGenerator(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        dropout=dropout,
    ).to(device)

    discriminator = PatchDiscriminator(
        in_channels=in_channels,
        base_channels=base_channels,
    ).to(device)

    g_optimizer, d_optimizer = build_optimizers(generator, discriminator, cfg)

    adv_criterion = nn.BCEWithLogitsLoss()
    recon_criterion = nn.L1Loss()
    edge_criterion = SobelEdgeLoss().to(device)

    g_scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))
    d_scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))

    start_epoch = 0
    best_metric = -float("inf")
    epochs_without_improvement = 0

    if args.resume is not None:
        ckpt = load_checkpoint(
            checkpoint_path=to_abs_path(args.resume),
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            g_scaler=g_scaler,
            d_scaler=d_scaler,
            map_location=device.type,
        )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_metric = float(ckpt.get("best_metric", -float("inf")))
        print(f"[INFO] Resume edildi. Start epoch = {start_epoch}")

    history_rows: List[Dict[str, Any]] = []

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        train_stats = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            loader=train_loader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            adv_criterion=adv_criterion,
            recon_criterion=recon_criterion,
            edge_criterion=edge_criterion,
            device=device,
            g_scaler=g_scaler,
            d_scaler=d_scaler,
            amp_enabled=amp_enabled,
            lambda_adv=lambda_adv,
            lambda_l1=lambda_l1,
            lambda_edge=lambda_edge,
        )

        val_stats = evaluate_generator(
            generator=generator,
            loader=val_loader,
            recon_criterion=recon_criterion,
            edge_criterion=edge_criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        # checkpoint metric seçimi:
        # Eğer AUROC hesaplanabiliyorsa onu maximize et,
        # aksi halde recon_l1 minimize et.
        current_metric = val_stats["image_auroc"]
        if math.isnan(current_metric):
            current_metric = -val_stats["recon_l1"]

        is_best = current_metric > best_metric
        if is_best:
            best_metric = float(current_metric)
            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                generator=generator,
                discriminator=discriminator,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                g_scaler=g_scaler,
                d_scaler=d_scaler,
                epoch=epoch,
                best_metric=best_metric,
                cfg=cfg,
            )

            val_rows = []
            for pid, path, label, score in zip(
                val_stats["patient_ids"],
                val_stats["image_paths"],
                val_stats["labels"].tolist(),
                val_stats["scores"].tolist(),
            ):
                val_rows.append(
                    {
                        "patient_id": pid,
                        "image_path": path,
                        "label": int(label),
                        "anomaly_score": float(score),
                    }
                )
            save_csv(val_rows, scores_dir / "best_val_scores.csv")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            g_scaler=g_scaler,
            d_scaler=d_scaler,
            epoch=epoch,
            best_metric=best_metric,
            cfg=cfg,
        )

        if (epoch + 1) % sample_every == 0:
            if val_stats["sample_x"] is not None:
                save_reconstruction_panel(
                    x=val_stats["sample_x"],
                    recon=val_stats["sample_recon"],
                    anomaly_map=val_stats["sample_map"],
                    save_path=recon_dir / f"epoch_{epoch + 1:03d}.png",
                    max_items=8,
                )

        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "time_sec": float(epoch_time),
            "train_g_loss": float(train_stats["g_loss"]),
            "train_d_loss": float(train_stats["d_loss"]),
            "train_recon_l1": float(train_stats["recon_l1"]),
            "train_edge_loss": float(train_stats["edge_loss"]),
            "train_adv_loss": float(train_stats["adv_loss"]),
            "val_recon_l1": float(val_stats["recon_l1"]),
            "val_edge_l1": float(val_stats["edge_l1"]),
            "val_image_auroc": float(val_stats["image_auroc"]) if not math.isnan(val_stats["image_auroc"]) else np.nan,
            "val_image_ap": float(val_stats["image_ap"]) if not math.isnan(val_stats["image_ap"]) else np.nan,
            "is_best": int(is_best),
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_dir / "history.csv", index=False)

        print(
            f"[Epoch {epoch + 1:03d}/{epochs:03d}] "
            f"G={train_stats['g_loss']:.4f} "
            f"D={train_stats['d_loss']:.4f} "
            f"val_l1={val_stats['recon_l1']:.4f} "
            f"val_auc={val_stats['image_auroc'] if not math.isnan(val_stats['image_auroc']) else 'nan'} "
            f"val_ap={val_stats['image_ap'] if not math.isnan(val_stats['image_ap']) else 'nan'}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"[INFO] Early stopping tetiklendi. patience={early_stopping_patience}")
            break

    best_ckpt_path = checkpoint_dir / "best.pt"
    if not best_ckpt_path.exists():
        raise FileNotFoundError("GAN için best.pt oluşmadı.")

    load_checkpoint(
        checkpoint_path=best_ckpt_path,
        generator=generator,
        discriminator=None,
        g_optimizer=None,
        d_optimizer=None,
        g_scaler=None,
        d_scaler=None,
        map_location=device.type,
    )

    # Final val
    final_val = evaluate_generator(
        generator=generator,
        loader=val_loader,
        recon_criterion=recon_criterion,
        edge_criterion=edge_criterion,
        device=device,
        amp_enabled=amp_enabled,
    )

    # Final test
    final_test = evaluate_generator(
        generator=generator,
        loader=test_loader,
        recon_criterion=recon_criterion,
        edge_criterion=edge_criterion,
        device=device,
        amp_enabled=amp_enabled,
    )

    # Test score CSV
    test_rows = []
    for pid, path, label, score in zip(
        final_test["patient_ids"],
        final_test["image_paths"],
        final_test["labels"].tolist(),
        final_test["scores"].tolist(),
    ):
        test_rows.append(
            {
                "patient_id": pid,
                "image_path": path,
                "label": int(label),
                "anomaly_score": float(score),
            }
        )
    save_csv(test_rows, scores_dir / "test_scores.csv")

    # Final reconstruction panel
    if final_test["sample_x"] is not None:
        save_reconstruction_panel(
            x=final_test["sample_x"],
            recon=final_test["sample_recon"],
            anomaly_map=final_test["sample_map"],
            save_path=recon_dir / "final_test_panel.png",
            max_items=8,
        )

    summary = {
        "run_name": run_name,
        "best_checkpoint": str(best_ckpt_path),
        "best_metric": float(best_metric),
        "num_train_normal": int(len(normal_train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "device": str(device),
        "val_metrics": {
            "recon_l1": float(final_val["recon_l1"]),
            "edge_l1": float(final_val["edge_l1"]),
            "image_auroc": float(final_val["image_auroc"]) if not math.isnan(final_val["image_auroc"]) else None,
            "image_ap": float(final_val["image_ap"]) if not math.isnan(final_val["image_ap"]) else None,
        },
        "test_metrics": {
            "recon_l1": float(final_test["recon_l1"]),
            "edge_l1": float(final_test["edge_l1"]),
            "image_auroc": float(final_test["image_auroc"]) if not math.isnan(final_test["image_auroc"]) else None,
            "image_ap": float(final_test["image_ap"]) if not math.isnan(final_test["image_ap"]) else None,
        },
    }

    save_json(summary, metrics_dir / "summary.json")
    pd.DataFrame(
        [
            {
                "recon_l1": float(final_val["recon_l1"]),
                "edge_l1": float(final_val["edge_l1"]),
                "image_auroc": float(final_val["image_auroc"]) if not math.isnan(final_val["image_auroc"]) else np.nan,
                "image_ap": float(final_val["image_ap"]) if not math.isnan(final_val["image_ap"]) else np.nan,
            }
        ]
    ).to_csv(metrics_dir / "val_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "recon_l1": float(final_test["recon_l1"]),
                "edge_l1": float(final_test["edge_l1"]),
                "image_auroc": float(final_test["image_auroc"]) if not math.isnan(final_test["image_auroc"]) else np.nan,
                "image_ap": float(final_test["image_ap"]) if not math.isnan(final_test["image_ap"]) else np.nan,
            }
        ]
    ).to_csv(metrics_dir / "test_metrics.csv", index=False)

    print("\n[INFO] GAN eğitimi tamamlandı.")
    print(f"[INFO] En iyi checkpoint : {best_ckpt_path}")
    print(f"[INFO] Test scores       : {scores_dir / 'test_scores.csv'}")
    print(f"[INFO] Summary          : {metrics_dir / 'summary.json'}")
    print(f"[INFO] Recon panel      : {recon_dir / 'final_test_panel.png'}")


if __name__ == "__main__":
    main()


