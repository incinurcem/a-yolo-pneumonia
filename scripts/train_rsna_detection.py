import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import math

# Albumentations update uyarısını sustur
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, dataloader

from dataset_rsna_detection import (
    RSNADetectionConfig,
    create_train_val_dataloaders,
)

try:
    from model_diffusion_guided_deformable_detr import DiffusionGuidedDeformableDETR
except ImportError:
    from diffusion_guided_deformable_detr import DiffusionGuidedDeformableDETR

from matcher import HungarianMatcher
from criterion import SetCriterion
from box_ops import box_cxcywh_to_xyxy, generalized_box_iou


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class TrainConfig:
    # paths
    train_csv: str = "/content/drive/MyDrive/Spring Semester/deep learning project/diffusion_guided_detr_data/metadata/train_master.csv"
    val_csv: str = "/content/drive/MyDrive/Spring Semester/deep learning project/diffusion_guided_detr_data/metadata/val_master.csv"
    output_dir: str = "/content/drive/MyDrive/Spring Semester/deep learning project/outputs/diffusion_guided_detr/run3"

    # resume
    resume_checkpoint: Optional[str] = None #"/content/drive/MyDrive/Spring Semester/deep learning project/outputs/diffusion_guided_detr/run2/checkpoints/epoch_032.pth"
    resume_history: bool = True
    strict_resume: bool = True

    # dataset / image
    image_size: int = 384
    batch_size: int = 64
    num_workers: int = 8
    apply_clahe: bool = False
    norm_mode: str = "imagenet"
    to_3channel: bool = False

    # training
    seed: int = 42
    epochs: int = 15
    lr: float = 2e-4
    lr_backbone: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    amp: bool = True

    # model
    # GÜNCELLEME: DETR Standardı -> 0: lesion, 1: background (no-object)
    num_classes: int = 2   
    backbone_name: str = "swin_tiny_patch4_window7_224"
    backbone_pretrained: bool = True
    num_queries: int = 100
    hidden_dim: int = 256
    num_feature_levels: int = 4
    fusion_mode: str = "hybrid"
    decoder_layers: int = 6
    encoder_layers: int = 2
    n_heads: int = 8
    n_points: int = 4

    # matcher / criterion
    cost_class: float = 5.0
    cost_bbox: float = 2.0
    cost_giou: float = 1.0

    loss_ce_weight: float = 2.0
    loss_bbox_weight: float = 5.0
    loss_giou_weight: float = 2.0
    eos_coef: float = 0.1 # FP'yi azaltmak için biraz artırıldı

    # scheduler
    min_lr: float = 1e-6
    warmup_epochs: int = 5

    # monitoring
    score_thresh: float = 0.05 # Çok düşük threshold eval'i yavaşlatabilir
    iou_thresh: float = 0.5
    save_every: int = 1
    metric_for_best: str = "val_loss"

    # logging
    debug_val_first_batch: bool = True
    print_epoch_header: bool = True


# ============================================================
# 2. UTILITIES
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json_if_exists(path: str, default=None):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    config: TrainConfig,
):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "config": asdict(config),
    }
    torch.save(ckpt, path)


def validate_paths(cfg: TrainConfig):
    if not os.path.exists(cfg.train_csv):
        raise FileNotFoundError(f"Train CSV not found: {cfg.train_csv}")
    if not os.path.exists(cfg.val_csv):
        raise FileNotFoundError(f"Val CSV not found: {cfg.val_csv}")


def load_checkpoint_for_resume(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    strict: bool = True,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    last_finished_epoch = int(ckpt.get("epoch", 0))
    start_epoch = last_finished_epoch + 1
    best_metric = ckpt.get("best_metric", float("inf"))
    best_epoch = ckpt.get("best_epoch", last_finished_epoch)

    return start_epoch, best_metric, best_epoch, ckpt


# ============================================================
# 3. DATASET CONFIG BRIDGE
# ============================================================

def build_dataset_config(cfg: TrainConfig) -> RSNADetectionConfig:
    return RSNADetectionConfig(
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        apply_clahe=cfg.apply_clahe,
        norm_mode=cfg.norm_mode,
        to_3channel=cfg.to_3channel,
    )


import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

def collate_fn(batch):
    """
    DETR için özel paketleyici: 
    Resimleri stack eder, hedefleri (targets) ve meta verileri liste olarak bırakır.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    metas = [item[2] for item in batch] if len(batch[0]) > 2 else None
    
    if metas is not None:
        return images, targets, metas
    return images, targets

def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    ds_cfg = build_dataset_config(cfg)
    
    # 1. Datasetleri oluştur
    train_ds, val_ds, _, _ = create_train_val_dataloaders(
        train_csv=cfg.train_csv,
        val_csv=cfg.val_csv,
        cfg=ds_cfg
    )

    # 2. Sampler Ağırlıklarını Hesapla
    train_df = pd.read_csv(cfg.train_csv)
    targets = train_df['target'].values 
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weight[t] for t in targets])).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    # 3. Train Loader (DÜZELTİLDİ: collate_fn eklendi)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler, 
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn # Artık kutular birbirine karışmayacak
    )

    # 4. Val Loader (DÜZELTİLDİ: collate_fn eklendi)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn # Val kısmında da aynı paketleyici lazım
    )

    return train_loader, val_loader

# ============================================================
# 4. MATCHER / CRITERION / MODEL / OPTIMIZER
# ============================================================

def build_matcher_criterion(cfg: TrainConfig):
    matcher = HungarianMatcher(
        cost_class=cfg.cost_class,
        cost_bbox=cfg.cost_bbox,
        cost_giou=cfg.cost_giou,
    )

    # Ana katman (final prediction) ağırlıkları
    weight_dict = {
        "loss_ce": cfg.loss_ce_weight,
        "loss_bbox": cfg.loss_bbox_weight,
        "loss_giou": cfg.loss_giou_weight,
    }

    # YARDIMCI KATMANLAR (Deformable DETR'da genelde 5 yardımcı katman olur)
    # GÜNCELLEME: Val Loss'un 10'da takılmaması için bu katmanların ağırlığını 0.5 yapıyoruz.
    for i in range(cfg.decoder_layers - 1):
        weight_dict.update({
            f"loss_ce_{i}": cfg.loss_ce_weight * 0.5,
            f"loss_bbox_{i}": cfg.loss_bbox_weight * 0.5,
            f"loss_giou_{i}": cfg.loss_giou_weight * 0.5,
        })

    criterion = SetCriterion(
        num_classes=cfg.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=cfg.eos_coef,
        losses=["labels", "boxes", "cardinality"],
    )
    return matcher, criterion


def build_model(cfg: TrainConfig, criterion: nn.Module, device: torch.device):
    model = DiffusionGuidedDeformableDETR(
        num_classes=cfg.num_classes,
        image_size=cfg.image_size,
        num_queries=cfg.num_queries,
        hidden_dim=cfg.hidden_dim,
        num_feature_levels=cfg.num_feature_levels,
        backbone_name=cfg.backbone_name,
        backbone_pretrained=cfg.backbone_pretrained,
        fusion_mode=cfg.fusion_mode,
        decoder_layers=cfg.decoder_layers,
        encoder_layers=cfg.encoder_layers,
        n_heads=cfg.n_heads,
        n_points=cfg.n_points,
        criterion=criterion,
    ).to(device)
    return model


def build_optimizer(cfg: TrainConfig, model: nn.Module):
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": other_params, "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )

    def lr_lambda(epoch_idx: int):
        if epoch_idx < cfg.warmup_epochs:
            return float(epoch_idx + 1) / float(max(cfg.warmup_epochs, 1))

        progress = float(epoch_idx - cfg.warmup_epochs) / float(max(cfg.epochs - cfg.warmup_epochs, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        min_factor_backbone = cfg.min_lr / max(cfg.lr_backbone, 1e-12)
        min_factor_other = cfg.min_lr / max(cfg.lr, 1e-12)
        min_factor = min(min_factor_backbone, min_factor_other)

        return max(cosine, min_factor)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


# ============================================================
# 5. OUTPUT HANDLING
# ============================================================

def extract_output_value(outputs: Any, key: str, default=None):
    if isinstance(outputs, dict):
        return outputs.get(key, default)
    return getattr(outputs, key, default)


def extract_losses(outputs: Any) -> Dict[str, torch.Tensor]:
    if isinstance(outputs, dict):
        if "losses" in outputs and isinstance(outputs["losses"], dict):
            return outputs["losses"]

        candidate = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor) and k.startswith("loss"):
                candidate[k] = v
        return candidate

    losses = getattr(outputs, "losses", None)
    if isinstance(losses, dict):
        return losses

    candidate = {}
    for k in dir(outputs):
        if k.startswith("loss"):
            v = getattr(outputs, k)
            if torch.is_tensor(v):
                candidate[k] = v
    return candidate


def extract_predictions(outputs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_logits = extract_output_value(outputs, "pred_logits", None)
    pred_boxes = extract_output_value(outputs, "pred_boxes", None)

    if pred_logits is None or pred_boxes is None:
        raise ValueError("Model output must contain pred_logits and pred_boxes.")

    return pred_logits, pred_boxes


# ============================================================
# 6. METRICS
# ============================================================

@torch.no_grad()
def postprocess_predictions(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    score_thresh: float = 0.3,
    num_classes: int = 2
) -> List[Dict[str, torch.Tensor]]:
    probs = pred_logits.softmax(-1)

    results = []
    bg_idx = num_classes - 1 # Arka plan artık son indeks (1)

    for b in range(pred_logits.shape[0]):
        scores, labels = probs[b].max(-1)
        # GÜNCELLEME: background (bg_idx) olmayanları tut
        keep = (labels != bg_idx) & (scores >= score_thresh)

        results.append({
            "scores": scores[keep],
            "labels": labels[keep],
            "boxes": pred_boxes[b][keep],
        })

    return results


@torch.no_grad()
def compute_batch_detection_stats(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    iou_thresh: float = 0.5,
    score_thresh: float = 0.3,
    num_classes: int = 2
) -> Dict[str, float]:
    preds = postprocess_predictions(
        pred_logits=pred_logits,
        pred_boxes=pred_boxes,
        score_thresh=score_thresh,
        num_classes=num_classes
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious = []

    for pred, tgt in zip(preds, targets):
        pred_boxes_ = pred["boxes"]
        pred_labels = pred["labels"]

        tgt_boxes = tgt["boxes"]
        tgt_labels = tgt["labels"] # RSNA etiketleri burada 1 gelir, ama matcher bunu düzelttiği sürece sorun yok.

        if len(tgt_boxes) == 0:
            total_fp += len(pred_boxes_)
            continue

        if len(pred_boxes_) == 0:
            total_fn += len(tgt_boxes)
            continue

        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        iou_mat = generalized_box_iou(pred_xyxy, tgt_xyxy)

        used_tgt = set()
        for i in range(len(pred_boxes_)):
            # GÜNCELLEME: targets'dan gelen label genelde 1'dir. 
            # Bizim modelimiz 0 tahmin ettiği için buradaki eşleşmeyi esnek tutuyoruz
            # RSNA pneumonia binary olduğu için sınıfa bakmaksızın en iyi box'ı eşleştirebiliriz
            scores = iou_mat[i] 
            best_val, best_j = scores.max(dim=0)

            if best_val.item() >= iou_thresh and int(best_j.item()) not in used_tgt:
                total_tp += 1
                used_tgt.add(int(best_j.item()))
                matched_ious.append(float(best_val.item()))
            else:
                total_fp += 1

        total_fn += len(tgt_boxes) - len(used_tgt)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
    }


# ============================================================
# 7. TRAIN HELPERS
# ============================================================

def move_batch_to_device(
    images: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    device: torch.device
):
    images = images.to(device, non_blocking=True)

    moved_targets = []
    for t in targets:
        moved_t = {}
        for k, v in t.items():
            if torch.is_tensor(v):
                moved_t[k] = v.to(device, non_blocking=True)
            else:
                moved_t[k] = v
        moved_targets.append(moved_t)

    return images, moved_targets


def compute_total_loss(loss_dict: Dict[str, torch.Tensor], weight_dict: Dict[str, float]) -> torch.Tensor:
    if len(loss_dict) == 0:
        return torch.tensor(0.0)

    device = next(iter(loss_dict.values())).device
    total_loss = torch.tensor(0.0, device=device)
    for k, v in loss_dict.items():
        if k in weight_dict:
            total_loss = total_loss + v * weight_dict[k]
    return total_loss


# ============================================================
# 8. ONE EPOCH TRAIN / VALIDATE
# ============================================================

import sys
def train_one_epoch(
    model: nn.Module,
    criterion: SetCriterion,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()

    # Metrik takibi için sayaçlar
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    bbox_meter = AverageMeter()
    giou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    miou_meter = AverageMeter()

    use_amp = cfg.amp and device.type == "cuda"
    num_steps = len(loader)
    
    # Gradyan Biriktirme Adımı (Config'den alıyoruz, yoksa varsayılan 4)
    # 32 (Physical) * 4 (Accumulation) = 128 (Effective Batch Size)
    accum_steps = getattr(cfg, "accumulation_steps", 4)
    
    # Gradyanları en başta bir kez sıfırlıyoruz
    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets, metas) in enumerate(loader):
        # 1. Veriyi GPU'ya taşı
        images, targets = move_batch_to_device(images, targets, device)

        # 2. İleri Geçiş (Forward Pass)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, targets)
            loss_dict = extract_losses(outputs)
            
            # Ana kayıp hesaplaması
            total_loss = compute_total_loss(loss_dict, criterion.weight_dict)
            
            # KRİTİK: Kaybı biriktirme adımına bölüyoruz. 
            # Çünkü gradyanlar toplandığı için ortalamayı korumamız lazım.
            loss_scaled = total_loss / accum_steps

        # 3. Geri Yayılım (Backward Pass)
        if scaler is not None and use_amp:
            # Scaler ile gradyanları hesapla (ama henüz step yapma)
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # 4. Optimizasyon Adımı (Sadece accum_steps'de bir veya epoch sonunda)
        if (step + 1) % accum_steps == 0 or (step + 1) == num_steps:
            if scaler is not None and use_amp:
                # Gradyan Kırpma (Gradient Clipping)
                if cfg.grad_clip is not None and cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                
                # Ağırlıkları güncelle
                scaler.step(optimizer)
                scaler.update()
            else:
                if cfg.grad_clip is not None and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            
            # Gradyanları temizle (Bir sonraki biriktirme döngüsü için)
            optimizer.zero_grad(set_to_none=True)

        # 5. Metrikleri Güncelle (Loglama için bölünmemiş total_loss'u kullanıyoruz)
        with torch.no_grad():
            pred_logits, pred_boxes = extract_predictions(outputs)
            batch_size = images.size(0)
            
            loss_meter.update(total_loss.item(), batch_size)
            ce_meter.update(loss_dict.get("loss_ce", torch.tensor(0.0, device=device)).item(), batch_size)
            bbox_meter.update(loss_dict.get("loss_bbox", torch.tensor(0.0, device=device)).item(), batch_size)
            giou_meter.update(loss_dict.get("loss_giou", torch.tensor(0.0, device=device)).item(), batch_size)

            stats = compute_batch_detection_stats(
                pred_logits=pred_logits.detach(),
                pred_boxes=pred_boxes.detach(),
                targets=targets,
                iou_thresh=cfg.iou_thresh,
                score_thresh=cfg.score_thresh,
                num_classes=cfg.num_classes
            )
            precision_meter.update(stats["precision"], batch_size)
            recall_meter.update(stats["recall"], batch_size)
            miou_meter.update(stats.get("mean_iou", 0.0), batch_size)

        # --- CANLI GÜNCELLEME ---
        print(
            f"\r[TRAIN] Epoch {epoch} | Step {step + 1}/{num_steps} | "
            f"Loss: {loss_meter.avg:.4f} | Prec: {precision_meter.avg:.4f} | Rec: {recall_meter.avg:.4f}",
            end="",
            flush=True
        )

    print() # Epoch bittiğinde alt satıra geç

    return {
        "train_loss": loss_meter.avg,
        "train_loss_ce": ce_meter.avg,
        "train_loss_bbox": bbox_meter.avg,
        "train_loss_giou": giou_meter.avg,
        "train_precision": precision_meter.avg,
        "train_recall": recall_meter.avg,
        "train_mean_iou": miou_meter.avg,
    }
@torch.no_grad()
def debug_batch_topk_scores(pred_logits: torch.Tensor, k: int = 10):
    probs = pred_logits.softmax(-1)
    if probs.shape[-1] < 2:
        return None
    # GÜNCELLEME: Foreground artık 0. indeks
    fg_scores = probs[..., 0]
    topk_vals, _ = torch.topk(fg_scores.reshape(-1), k=min(k, fg_scores.numel()))
    return topk_vals.detach().cpu().tolist()


import sys

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    criterion: SetCriterion,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    bbox_meter = AverageMeter()
    giou_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    miou_meter = AverageMeter()

    bg_idx = cfg.num_classes - 1
    num_steps = len(loader)

    for step, (images, targets, metas) in enumerate(loader):
        # Veriyi GPU'ya taşı
        images, targets = move_batch_to_device(images, targets, device)

        # İleri Geçiş
        outputs = model(images, targets)
        loss_dict = extract_losses(outputs)
        total_loss = compute_total_loss(loss_dict, criterion.weight_dict)
        pred_logits, pred_boxes = extract_predictions(outputs)

        # İlk batch için detaylı debug bilgisi (Progress bar'ın üzerinde görünür)
        if cfg.debug_val_first_batch and step == 0:
            probs = pred_logits.softmax(-1)
            max_scores, pred_labels = probs.max(-1)
            # Burada \n kullanarak detayları yeni satırlara yazıyoruz
            print(
                f"\n[DEBUG][VAL][Epoch {epoch:03d}] "
                f"max_score_mean={max_scores.mean().item():.4f} | "
                f"max_score_max={max_scores.max().item():.4f} | "
                f"foreground_ratio={(pred_labels != bg_idx).float().mean().item():.4f}"
            )
            topk_scores = debug_batch_topk_scores(pred_logits, k=10)
            print(f"[DEBUG][VAL][Epoch {epoch:03d}] top10_fg_scores={topk_scores}")

        # Metrikleri Güncelle
        batch_size = images.size(0)
        loss_meter.update(total_loss.item(), batch_size)
        ce_meter.update(loss_dict.get("loss_ce", torch.tensor(0.0, device=device)).item(), batch_size)
        bbox_meter.update(loss_dict.get("loss_bbox", torch.tensor(0.0, device=device)).item(), batch_size)
        giou_meter.update(loss_dict.get("loss_giou", torch.tensor(0.0, device=device)).item(), batch_size)

        stats = compute_batch_detection_stats(
            pred_logits=pred_logits.detach(),
            pred_boxes=pred_boxes.detach(),
            targets=targets,
            iou_thresh=cfg.iou_thresh,
            score_thresh=cfg.score_thresh,
            num_classes=cfg.num_classes
        )
        precision_meter.update(stats["precision"], batch_size)
        recall_meter.update(stats["recall"], batch_size)
        miou_meter.update(stats.get("mean_iou", 0.0), batch_size)

        # --- TEK SATIRDA CANLI GÜNCELLEME ---
        # \r karakteri imleci satır başına getirir, end="" ise alt satıra geçmeyi engeller.
        print(
            f"\r[VAL] Epoch {epoch} | Step {step + 1}/{num_steps} | "
            f"Loss: {loss_meter.avg:.4f} | Prec: {precision_meter.avg:.4f} | Rec: {recall_meter.avg:.4f}",
            end="",
            flush=True
        )

    # Val bittiğinde bir alt satıra geç ki bir sonraki epoch yazıları karışmasın
    print()

    return {
        "val_loss": loss_meter.avg,
        "val_loss_ce": ce_meter.avg,
        "val_loss_bbox": bbox_meter.avg,
        "val_loss_giou": giou_meter.avg,
        "val_precision": precision_meter.avg,
        "val_recall": recall_meter.avg,
        "val_mean_iou": miou_meter.avg,
    }


# ============================================================
# 9. MAIN TRAINING
# ============================================================

def main():
    cfg = TrainConfig()

    validate_paths(cfg)

    ensure_dir(cfg.output_dir)
    ensure_dir(os.path.join(cfg.output_dir, "checkpoints"))
    ensure_dir(os.path.join(cfg.output_dir, "logs"))

    config_path = os.path.join(cfg.output_dir, "config.json")
    history_path = os.path.join(cfg.output_dir, "logs", "history.json")
    summary_path = os.path.join(cfg.output_dir, "logs", "summary.json")

    save_json(asdict(cfg), config_path)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("Device:", device)
    print("=" * 100)

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    _, criterion = build_matcher_criterion(cfg)
    criterion = criterion.to(device)

    model = build_model(cfg, criterion, device)
    optimizer, scheduler = build_optimizer(cfg, model)

    use_amp = cfg.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    loaded_history = load_json_if_exists(history_path, default=None)
    if cfg.resume_history and loaded_history is not None and isinstance(loaded_history, dict):
        history = loaded_history.get("history", [])

    best_metric = float("inf") if cfg.metric_for_best == "val_loss" else -float("inf")
    best_epoch = -1
    start_epoch = 1

    if cfg.resume_checkpoint is not None:
        start_epoch, best_metric, best_epoch, ckpt = load_checkpoint_for_resume(
            checkpoint_path=cfg.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            strict=cfg.strict_resume,
        )

        print("=" * 100)
        print(f"Resume checkpoint : {cfg.resume_checkpoint}")
        print(f"Loaded epoch      : {ckpt.get('epoch', 'N/A')}")
        print(f"Start epoch       : {start_epoch}")
        print(f"Best metric       : {best_metric}")
        print(f"Best epoch        : {best_epoch}")
        print("=" * 100)

        if start_epoch > cfg.epochs:
            print("Checkpoint already reached or exceeded target epochs. Nothing to train.")
            return

        if len(history) > 0:
            history = [h for h in history if int(h.get("epoch", 0)) < start_epoch]

    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.print_epoch_header:
            print("=" * 100)
            print(f"Epoch {epoch}/{cfg.epochs}")
            print("=" * 100)

        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            epoch=epoch,
            cfg=cfg,
            scaler=scaler,
        )

        val_metrics = validate_one_epoch(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            epoch=epoch,
            cfg=cfg,
        )

        scheduler.step()

        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_other = optimizer.param_groups[1]["lr"]

        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr_backbone": current_lr_backbone,
            "lr_other": current_lr_other,
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"train_loss={epoch_metrics['train_loss']:.4f} | "
            f"val_loss={epoch_metrics['val_loss']:.4f} | "
            f"train_prec={epoch_metrics['train_precision']:.4f} | "
            f"train_rec={epoch_metrics['train_recall']:.4f} | "
            f"val_prec={epoch_metrics['val_precision']:.4f} | "
            f"val_rec={epoch_metrics['val_recall']:.4f} | "
            f"val_mIoU={epoch_metrics['val_mean_iou']:.4f} | "
            f"lr_backbone={epoch_metrics['lr_backbone']:.2e} | "
            f"lr_other={epoch_metrics['lr_other']:.2e}"
        )

        save_json({"history": history}, history_path)

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                path=os.path.join(cfg.output_dir, "checkpoints", f"epoch_{epoch:03d}.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                config=cfg,
            )

        current_metric = epoch_metrics[cfg.metric_for_best]
        is_better = (
            current_metric < best_metric
            if cfg.metric_for_best == "val_loss"
            else current_metric > best_metric
        )

        if is_better:
            best_metric = current_metric
            best_epoch = epoch
            save_checkpoint(
                path=os.path.join(cfg.output_dir, "checkpoints", "best.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                config=cfg,
            )
            print(f"New best model saved at epoch {epoch} with {cfg.metric_for_best}={best_metric:.6f}")

    save_checkpoint(
        path=os.path.join(cfg.output_dir, "checkpoints", "last.pth"),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=cfg.epochs,
        best_metric=best_metric,
        best_epoch=best_epoch,
        config=cfg,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_metric_name": cfg.metric_for_best,
        "best_metric_value": best_metric,
        "num_epochs": cfg.epochs,
        "resume_checkpoint": cfg.resume_checkpoint,
    }
    save_json(summary, summary_path)

    print("=" * 100)
    print("Training completed.")
    print("Best epoch :", best_epoch)
    print("Best metric:", cfg.metric_for_best, "=", best_metric)
    print("=" * 100)


if __name__ == "__main__":
    main()