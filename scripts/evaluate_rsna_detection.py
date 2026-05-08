# -*- coding: utf-8 -*-
"""
evaluate_rsna_detection.py

Enhanced evaluation script for:
Diffusion-Guided Deformable DETR on RSNA pneumonia detection

Major improvements over the previous version:
- Stronger image-level decision logic
- Multiple aggregation modes
- Min box area filtering
- Min positive box count rule
- Threshold sweep for image-level classification
- Better score distribution analysis
- Better medical metrics reporting
- Extra CSV/JSON outputs for post-analysis

Important:
This script improves evaluation faithfulness and reduces image-level false positives,
but it cannot magically fix a weak detector. It gives a much more realistic and
scientifically usable evaluation.

Author-aligned revision
"""

import os
import json
import ast
import math
import argparse
import importlib
from typing import Any, Dict, List, Tuple, Optional
import sys
import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# =========================================================
# BASIC UTILS
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def infer_column(df: pd.DataFrame, candidates: List[str], required: bool = True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def constant_pad_kwargs():
    return {
        "border_mode": cv2.BORDER_CONSTANT,
        "value": 0,
        "mask_value": 0,
    }


def compute_iou_xyxy(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(max(w - 1, 0))))
    y1 = max(0.0, min(float(y1), float(max(h - 1, 0))))
    x2 = max(0.0, min(float(x2), float(max(w, 0))))
    y2 = max(0.0, min(float(y2), float(max(h, 0))))
    return [x1, y1, x2, y2]


def box_xywh_to_xyxy(box):
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def cxcywh_to_xyxy_tensor(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def nms_numpy(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def softmax_np(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.maximum(np.sum(ex, axis=axis, keepdims=True), 1e-12)


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


# =========================================================
# PREPROCESSING
# =========================================================
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


def build_normalize(norm_mode: str, to_3channel: bool, max_pixel_value: float = 255.0):
    if to_3channel:
        if norm_mode == "imagenet":
            return A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=max_pixel_value
            )
        elif norm_mode == "minmax_01":
            return A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                max_pixel_value=max_pixel_value
            )
        else:
            raise ValueError(f"Unsupported norm_mode: {norm_mode}")
    else:
        if norm_mode == "imagenet":
            return A.Normalize(
                mean=(0.485,),
                std=(0.229,),
                max_pixel_value=max_pixel_value
            )
        elif norm_mode == "minmax_01":
            return A.Normalize(
                mean=(0.0,),
                std=(1.0,),
                max_pixel_value=max_pixel_value
            )
        else:
            raise ValueError(f"Unsupported norm_mode: {norm_mode}")


def get_eval_transform(image_size: int, norm_mode: str, to_3channel: bool):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                p=1.0,
                **constant_pad_kwargs()
            ),
            build_normalize(norm_mode=norm_mode, to_3channel=to_3channel),
        ]
    )


def compute_resize_pad_params(orig_h: int, orig_w: int, image_size: int) -> Dict[str, float]:
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid image size: h={orig_h}, w={orig_w}")

    scale = float(image_size) / float(max(orig_h, orig_w))
    resized_h = max(1, int(round(orig_h * scale)))
    resized_w = max(1, int(round(orig_w * scale)))

    pad_h_total = max(image_size - resized_h, 0)
    pad_w_total = max(image_size - resized_w, 0)

    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left

    return {
        "scale": scale,
        "resized_h": resized_h,
        "resized_w": resized_w,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }


def map_boxes_from_padded_to_original(
    boxes_xyxy: List[List[float]],
    orig_w: int,
    orig_h: int,
    image_size: int
) -> List[List[float]]:
    if len(boxes_xyxy) == 0:
        return []

    params = compute_resize_pad_params(orig_h=orig_h, orig_w=orig_w, image_size=image_size)
    scale = params["scale"]
    pad_left = params["pad_left"]
    pad_top = params["pad_top"]

    mapped = []
    for b in boxes_xyxy:
        x1, y1, x2, y2 = b

        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        mapped.append(clip_box_xyxy([x1, y1, x2, y2], orig_w, orig_h))

    return mapped


# =========================================================
# GT / CSV PARSING
# =========================================================
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
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return []


def resolve_image_path(row, patient_id_col, image_path_col, image_dir):
    if image_path_col is not None:
        p = str(row[image_path_col])
        if os.path.exists(p):
            return p

        if image_dir is not None:
            maybe = os.path.join(image_dir, os.path.basename(p))
            if os.path.exists(maybe):
                return maybe

        return p

    patient_id = str(row[patient_id_col])
    if image_dir is None:
        raise ValueError(
            "CSV does not contain image_path and --image_dir was not provided. "
            "Need one of them to resolve images."
        )

    candidates = [
        os.path.join(image_dir, f"{patient_id}.png"),
        os.path.join(image_dir, f"{patient_id}.jpg"),
        os.path.join(image_dir, f"{patient_id}.jpeg"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    return candidates[0]


def build_grouped_records(csv_path, image_dir=None):
    df = pd.read_csv(csv_path)

    patient_col = infer_column(df, ["patientId", "patient_id", "image_id", "id"])
    image_path_col = infer_column(df, ["image_path", "image_file", "filepath", "path", "file_path"], required=False)

    boxes_xywh_col = infer_column(df, ["boxes_xywh"], required=False)
    target_col = infer_column(df, ["target", "Target", "label", "class_id"], required=False)

    x_col = infer_column(df, ["x", "xmin", "X"], required=False)
    y_col = infer_column(df, ["y", "ymin", "Y"], required=False)
    w_col = infer_column(df, ["width", "w", "W"], required=False)
    h_col = infer_column(df, ["height", "h", "H"], required=False)

    grouped_records = []

    if boxes_xywh_col is not None:
        for _, row in df.iterrows():
            pid = str(row[patient_col])
            image_path = resolve_image_path(row, patient_col, image_path_col, image_dir)

            boxes_xywh = safe_parse_boxes_xywh(row.get(boxes_xywh_col, "[]"))
            boxes = [box_xywh_to_xyxy(box) for box in boxes_xywh if len(box) == 4]

            if target_col is not None:
                label = safe_int(row.get(target_col, 0), 0)
                if label < 0:
                    label = int(len(boxes) > 0)
            else:
                label = int(len(boxes) > 0)

            grouped_records.append({
                "patient_id": pid,
                "image_path": image_path,
                "boxes": boxes,
                "label": int(label),
            })

        return grouped_records

    for pid, g in df.groupby(patient_col):
        first_row = g.iloc[0]
        image_path = resolve_image_path(first_row, patient_col, image_path_col, image_dir)

        boxes = []
        if x_col and y_col and w_col and h_col:
            for _, row in g.iterrows():
                x = safe_float(row[x_col], np.nan)
                y = safe_float(row[y_col], np.nan)
                w = safe_float(row[w_col], np.nan)
                h = safe_float(row[h_col], np.nan)

                if (
                    not np.isnan(x) and
                    not np.isnan(y) and
                    not np.isnan(w) and
                    not np.isnan(h) and
                    w > 0 and h > 0
                ):
                    boxes.append(box_xywh_to_xyxy([x, y, w, h]))

        if target_col is not None:
            tv = pd.to_numeric(g[target_col], errors="coerce").fillna(0).values
            label = int(np.max(tv) > 0)
        else:
            label = int(len(boxes) > 0)

        grouped_records.append({
            "patient_id": str(pid),
            "image_path": image_path,
            "boxes": boxes,
            "label": label,
        })

    return grouped_records


# =========================================================
# DATASET
# =========================================================
class RSNADetectionEvalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_dir: Optional[str],
        image_size: int,
        apply_clahe: bool,
        norm_mode: str,
        to_3channel: bool,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.records = build_grouped_records(csv_path=csv_path, image_dir=image_dir)
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.norm_mode = norm_mode
        self.to_3channel = to_3channel
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.transform = get_eval_transform(
            image_size=image_size,
            norm_mode=norm_mode,
            to_3channel=to_3channel,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        image = read_png_grayscale(rec["image_path"])
        image = apply_clahe_if_needed(
            image_uint8=image,
            enabled=self.apply_clahe,
            clip_limit=self.clahe_clip_limit,
            tile_grid_size=self.clahe_tile_grid_size,
        )

        orig_h, orig_w = image.shape[:2]

        if self.to_3channel:
            image = grayscale_to_3channel(image)
        else:
            image = np.expand_dims(image, axis=-1)

        transformed = self.transform(image=image)
        image = transformed["image"]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        meta = {
            "patient_id": rec["patient_id"],
            "image_path": rec["image_path"],
            "orig_h": orig_h,
            "orig_w": orig_w,
            "gt_boxes": rec["boxes"],
            "label": int(rec["label"]),
        }

        return image, meta


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return images, metas


# =========================================================
# MODEL BUILD / LOAD
# =========================================================
def build_model_from_module(args):
    mod = importlib.import_module(args.model_module)

    candidate_names = []
    if args.model_builder:
        candidate_names.append(args.model_builder)

    candidate_names.extend([
        "build_rsna_detection_model",
        "build_model",
        "get_model",
        "create_model",
        "build_diffusion_guided_deformable_detr",
    ])

    for name in candidate_names:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            kwargs_candidates = [
                {
                    "num_classes": args.num_classes,
                    "image_size": args.image_size,
                    "num_queries": args.num_queries,
                    "hidden_dim": args.hidden_dim,
                    "num_feature_levels": args.num_feature_levels,
                    "backbone_name": args.backbone_name,
                    "backbone_pretrained": False,
                    "fusion_mode": args.fusion_mode,
                    "decoder_layers": args.decoder_layers,
                    "encoder_layers": args.encoder_layers,
                    "n_heads": args.n_heads,
                    "n_points": args.n_points,
                    "criterion": None,
                },
                {
                    "num_classes": args.num_classes,
                    "img_size": args.image_size,
                    "num_queries": args.num_queries,
                },
                {
                    "num_classes": args.num_classes,
                    "image_size": args.image_size,
                    "num_queries": args.num_queries,
                },
                {
                    "num_classes": args.num_classes,
                },
                {},
            ]

            for kwargs in kwargs_candidates:
                try:
                    model = fn(**kwargs)
                    return model
                except TypeError:
                    continue

    if hasattr(mod, "DiffusionGuidedDeformableDETR"):
        cls = getattr(mod, "DiffusionGuidedDeformableDETR")
        model = cls(
            num_classes=args.num_classes,
            image_size=args.image_size,
            num_queries=args.num_queries,
            hidden_dim=args.hidden_dim,
            num_feature_levels=args.num_feature_levels,
            backbone_name=args.backbone_name,
            backbone_pretrained=False,
            fusion_mode=args.fusion_mode,
            decoder_layers=args.decoder_layers,
            encoder_layers=args.encoder_layers,
            n_heads=args.n_heads,
            n_points=args.n_points,
            criterion=None,
        )
        return model

    raise AttributeError(
        f"Could not build model from module='{args.model_module}'. "
        f"Neither compatible builder nor DiffusionGuidedDeformableDETR class was found."
    )


def extract_state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "weights",
            "ema_state_dict",
            "net",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt

    raise ValueError("Could not extract a valid state_dict from checkpoint.")


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned[nk] = v
    return cleaned


def load_checkpoint_weights(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict_from_checkpoint(ckpt)
    state_dict = clean_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("=" * 100)
    print("Checkpoint loaded.")
    print("Missing keys   :", missing)
    print("Unexpected keys:", unexpected)
    print("=" * 100)
    return model


# =========================================================
# OUTPUT DECODING
# =========================================================
def unwrap_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs

    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        first = outputs[0]
        if isinstance(first, (dict, list)):
            return first

        for item in outputs:
            if isinstance(item, dict):
                if ("pred_logits" in item and "pred_boxes" in item) or \
                   ("boxes" in item and "scores" in item):
                    return item

        for item in outputs:
            if hasattr(item, "pred_logits") and hasattr(item, "pred_boxes"):
                return {
                    "pred_logits": getattr(item, "pred_logits"),
                    "pred_boxes": getattr(item, "pred_boxes"),
                }
            if hasattr(item, "boxes") and hasattr(item, "scores"):
                out = {
                    "boxes": getattr(item, "boxes"),
                    "scores": getattr(item, "scores"),
                }
                if hasattr(item, "labels"):
                    out["labels"] = getattr(item, "labels")
                return out

    if hasattr(outputs, "pred_logits") and hasattr(outputs, "pred_boxes"):
        return {
            "pred_logits": getattr(outputs, "pred_logits"),
            "pred_boxes": getattr(outputs, "pred_boxes"),
        }

    if hasattr(outputs, "boxes") and hasattr(outputs, "scores"):
        out = {
            "boxes": getattr(outputs, "boxes"),
            "scores": getattr(outputs, "scores"),
        }
        if hasattr(outputs, "labels"):
            out["labels"] = getattr(outputs, "labels")
        return out

    if hasattr(outputs, "keys"):
        try:
            keys = list(outputs.keys())
            if "pred_logits" in keys and "pred_boxes" in keys:
                return {
                    "pred_logits": outputs["pred_logits"],
                    "pred_boxes": outputs["pred_boxes"],
                }
            if "boxes" in keys and "scores" in keys:
                out = {
                    "boxes": outputs["boxes"],
                    "scores": outputs["scores"],
                }
                if "labels" in keys:
                    out["labels"] = outputs["labels"]
                return out
        except Exception:
            pass

    return outputs

@torch.no_grad()
def decode_model_outputs(
    outputs,
    batch_size,
    image_size,
    score_thresh=0.05,
    nms_thresh=0.5,
    debug=False,
    debug_batches=1,
):
    """
    RSNA Pneumonia Detection için optimize edilmiş decoder.
    Yeni İndeks Standardı:
    - Index 0: Lesion (Pneumonia) -> Skor buradan alınır.
    - Index 1: Background (No-object) -> Görmezden gelinir.
    """

    outputs = unwrap_outputs(outputs)
    decoded = []

    # --- İÇ YARDIMCI FONKSİYONLAR ---
    def _to_numpy(x, dtype=None):
        if x is None: return None
        if torch.is_tensor(x): x = x.detach().cpu().numpy()
        else: x = np.asarray(x)
        if dtype is not None: x = x.astype(dtype)
        return x

    def _sanitize_boxes_scores_labels(boxes, scores, labels):
        boxes = _to_numpy(boxes, dtype=np.float32)
        scores = _to_numpy(scores, dtype=np.float32)
        labels = _to_numpy(labels, dtype=np.int64)

        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        n = min(len(boxes), len(scores), len(labels))
        return boxes[:n], scores[:n], labels[:n]

    def _finalize_single_prediction(boxes, scores, labels):
        boxes, scores, labels = _sanitize_boxes_scores_labels(boxes, scores, labels)
        if len(scores) == 0:
            return {"boxes": [], "scores": [], "labels": []}

        # Threshold filtreleme
        keep = scores >= float(score_thresh)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if len(scores) == 0:
            return {"boxes": [], "scores": [], "labels": []}

        # Kutuları resim boyutuna kırp (clip)
        clipped_boxes = []
        for b in boxes:
            clipped_boxes.append(clip_box_xyxy(b.tolist(), image_size, image_size))
        boxes = np.asarray(clipped_boxes, dtype=np.float32)

        # NMS (Üst üste binenleri temizle)
        if len(boxes) > 0:
            keep_nms = nms_numpy(boxes, scores, iou_thresh=nms_thresh)
            boxes, scores, labels = boxes[keep_nms], scores[keep_nms], labels[keep_nms]

        return {
            "boxes": boxes.tolist(),
            "scores": scores.astype(np.float32).tolist(),
            "labels": labels.astype(np.int64).tolist(),
        }

    # =====================================================
    # CASE 1: torchvision style -> list[dict]
    # =====================================================
    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
        for out in outputs:
            decoded.append(_finalize_single_prediction(out.get("boxes"), out.get("scores"), out.get("labels")))
        return decoded

    # =====================================================
    # CASE 2: DETR style -> dict(pred_logits, pred_boxes)
    # =====================================================
    if isinstance(outputs, dict) and "pred_logits" in outputs and "pred_boxes" in outputs:
        pred_logits = outputs["pred_logits"] # [B, Q, C]
        pred_boxes = outputs["pred_boxes"]   # [B, Q, 4]

        B, Q, C = pred_logits.shape
        
        for b in range(B):
            logits_b = pred_logits[b]
            boxes_b = pred_boxes[b]

            # 0. indeks lezyon, 1. indeks arka plan olduğu için:
            probs = torch.softmax(logits_b, dim=-1)
            scores = probs[:, 0] # Sadece lezyon (foreground) olasılığını al
            labels = torch.zeros_like(scores, dtype=torch.long) # Tüm tahminler lezyon adayı (Label 0)

            # Koordinat dönüşümü: cxcywh -> xyxy
            boxes_xyxy = cxcywh_to_xyxy_tensor(boxes_b)

            # Normalize [0,1] kutuları piksel boyutuna çek
            if boxes_xyxy.numel() > 0:
                boxes_xyxy[:, [0, 2]] *= float(image_size)
                boxes_xyxy[:, [1, 3]] *= float(image_size)

            decoded.append(_finalize_single_prediction(boxes_xyxy, scores, labels))
        return decoded

    # =====================================================
    # CASE 3: Direct dict with batched boxes/scores
    # =====================================================
    if isinstance(outputs, dict) and "boxes" in outputs and "scores" in outputs:
        for i in range(batch_size):
            decoded.append(_finalize_single_prediction(outputs["boxes"][i], outputs["scores"][i], outputs.get("labels", [None]*batch_size)[i]))
        return decoded

    raise ValueError("Bilinmeyen model çıktı formatı!")
# =========================================================
# MATCHING / AP
# =========================================================
def match_predictions_to_gts(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    if len(gt_boxes) == 0:
        return [
            {"score": float(s), "is_tp": 0, "best_iou": 0.0, "matched_gt_idx": -1}
            for s in pred_scores
        ], 0

    order = np.argsort(-np.asarray(pred_scores))
    matched_gt = set()
    matches = []

    for idx in order:
        pbox = pred_boxes[idx]
        score = pred_scores[idx]

        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou_xyxy(pbox, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        is_tp = int(best_iou >= iou_thresh and best_gt_idx >= 0)
        if is_tp:
            matched_gt.add(best_gt_idx)

        matches.append({
            "score": float(score),
            "is_tp": is_tp,
            "best_iou": float(best_iou),
            "matched_gt_idx": int(best_gt_idx),
        })

    return matches, len(gt_boxes)


def compute_ap_from_ranked(all_matches, total_gt):
    if total_gt == 0:
        return 0.0, [], []

    ranked = sorted(all_matches, key=lambda x: x["score"], reverse=True)
    tp = np.array([m["is_tp"] for m in ranked], dtype=np.float32)
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / max(total_gt, 1e-8)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    return float(ap), precisions.tolist(), recalls.tolist()


def compute_detection_summary(per_image_results, iou_thresholds=None, score_threshold_for_summary=0.30):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 0.96, 0.05)

    aps = {}
    for thr in iou_thresholds:
        all_matches = []
        total_gt = 0

        for rec in per_image_results:
            matches, n_gt = match_predictions_to_gts(
                rec["pred_boxes"], rec["pred_scores"], rec["gt_boxes"], iou_thresh=thr
            )
            all_matches.extend(matches)
            total_gt += n_gt

        ap, _, _ = compute_ap_from_ranked(all_matches, total_gt)
        aps[round(float(thr), 2)] = float(ap)

    ap50 = aps.get(0.50, 0.0)
    ap75 = aps.get(0.75, 0.0)
    map5095 = float(np.mean(list(aps.values()))) if len(aps) > 0 else 0.0

    tp = 0
    fp = 0
    fn = 0
    matched_ious = []

    for rec in per_image_results:
        keep = [i for i, s in enumerate(rec["pred_scores"]) if s >= score_threshold_for_summary]
        pred_boxes_thr = [rec["pred_boxes"][i] for i in keep]
        pred_scores_thr = [rec["pred_scores"][i] for i in keep]

        matches, n_gt = match_predictions_to_gts(
            pred_boxes_thr, pred_scores_thr, rec["gt_boxes"], iou_thresh=0.50
        )

        tp_img = sum(m["is_tp"] for m in matches)
        fp_img = len(matches) - tp_img
        fn_img = n_gt - tp_img

        tp += tp_img
        fp += fp_img
        fn += fn_img

        for m in matches:
            if m["is_tp"]:
                matched_ious.append(m["best_iou"])

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

    return {
        "AP50": float(ap50),
        "AP75": float(ap75),
        "mAP50_95": float(map5095),
        "AP_by_IoU": aps,
        "detection_precision_at_score_thresh": float(precision),
        "detection_recall_at_score_thresh": float(recall),
        "detection_f1_at_score_thresh": float(f1),
        "mean_matched_iou": float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0,
        "median_matched_iou": float(np.median(matched_ious)) if len(matched_ious) > 0 else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


# =========================================================
# IMAGE-LEVEL AGGREGATION
# =========================================================
def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def filter_boxes_for_image_level(
    boxes,
    scores,
    labels=None,
    min_box_score=0.20,
    min_box_area=1024.0,
    max_boxes_for_image_score=20,
):
    boxes = list(boxes)
    scores = list(scores)
    labels = list(labels) if labels is not None else [1] * len(scores)

    kept = []
    for b, s, l in zip(boxes, scores, labels):
        area = box_area_xyxy(b)
        if float(s) >= float(min_box_score) and area >= float(min_box_area):
            kept.append({
                "box": b,
                "score": float(s),
                "label": int(l),
                "area": float(area),
            })

    kept = sorted(kept, key=lambda x: x["score"], reverse=True)
    if max_boxes_for_image_score is not None and max_boxes_for_image_score > 0:
        kept = kept[:max_boxes_for_image_score]

    return kept


def aggregate_image_score(
    kept_boxes,
    mode="topk_mean",
    topk=3,
):
    if len(kept_boxes) == 0:
        return 0.0, {
            "num_valid_boxes": 0,
            "max_valid_score": 0.0,
            "mean_valid_score": 0.0,
            "sum_valid_score": 0.0,
            "max_valid_area": 0.0,
        }

    scores = np.array([x["score"] for x in kept_boxes], dtype=np.float64)
    areas = np.array([x["area"] for x in kept_boxes], dtype=np.float64)

    max_valid_score = float(np.max(scores))
    mean_valid_score = float(np.mean(scores))
    sum_valid_score = float(np.sum(scores))
    max_valid_area = float(np.max(areas))

    k = min(int(topk), len(scores)) if topk > 0 else len(scores)
    top_scores = scores[:k]
    top_areas = areas[:k]

    if mode == "max":
        image_score = max_valid_score

    elif mode == "topk_mean":
        image_score = float(np.mean(top_scores))

    elif mode == "topk_sum":
        image_score = float(np.sum(top_scores))

    elif mode == "logsumexp":
        image_score = float(np.log(np.sum(np.exp(top_scores))) / max(k, 1))

    elif mode == "area_weighted":
        w = top_areas / np.maximum(np.sum(top_areas), 1e-8)
        image_score = float(np.sum(w * top_scores))

    elif mode == "score_area_hybrid":
        norm_areas = np.sqrt(np.maximum(top_areas, 1.0))
        norm_areas = norm_areas / np.maximum(np.sum(norm_areas), 1e-8)
        image_score = float(np.sum(norm_areas * top_scores))

    else:
        raise ValueError(f"Unsupported image aggregation mode: {mode}")

    details = {
        "num_valid_boxes": int(len(kept_boxes)),
        "max_valid_score": max_valid_score,
        "mean_valid_score": mean_valid_score,
        "sum_valid_score": sum_valid_score,
        "max_valid_area": max_valid_area,
    }
    return image_score, details


def decide_image_positive(
    kept_boxes,
    image_score,
    image_score_thresh=0.35,
    min_positive_boxes=1,
):
    num_valid = len(kept_boxes)
    pred = int((num_valid >= int(min_positive_boxes)) and (float(image_score) >= float(image_score_thresh)))
    return pred


def compute_image_level_outputs_for_record(
    pred_boxes,
    pred_scores,
    pred_labels,
    agg_mode="topk_mean",
    topk=3,
    min_box_score=0.20,
    min_box_area=1024.0,
    min_positive_boxes=1,
    image_score_thresh=0.35,
    max_boxes_for_image_score=20,
):
    kept = filter_boxes_for_image_level(
        boxes=pred_boxes,
        scores=pred_scores,
        labels=pred_labels,
        min_box_score=min_box_score,
        min_box_area=min_box_area,
        max_boxes_for_image_score=max_boxes_for_image_score,
    )

    image_score, agg_details = aggregate_image_score(
        kept_boxes=kept,
        mode=agg_mode,
        topk=topk,
    )

    image_pred = decide_image_positive(
        kept_boxes=kept,
        image_score=image_score,
        image_score_thresh=image_score_thresh,
        min_positive_boxes=min_positive_boxes,
    )

    return {
        "image_score": float(image_score),
        "image_pred": int(image_pred),
        "num_valid_boxes_for_image": int(len(kept)),
        "valid_boxes_for_image": kept,
        **agg_details,
    }


# =========================================================
# MEDICAL METRICS
# =========================================================
def compute_medical_metrics(y_true, y_score, y_pred_binary):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred_binary = np.asarray(y_pred_binary).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1e-8)
    sensitivity = tp / max(tp + fn, 1e-8)
    specificity = tn / max(tn + fp, 1e-8)
    precision_ppv = tp / max(tp + fp, 1e-8)
    npv = tn / max(tn + fn, 1e-8)
    f1 = (2 * precision_ppv * sensitivity) / max(precision_ppv + sensitivity, 1e-8)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    youden_j = sensitivity + specificity - 1.0

    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(precision_ppv),
        "npv": float(npv),
        "f1_score": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "youden_j": float(youden_j),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def summarize_score_distribution(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    def desc(arr):
        if len(arr) == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p05": None,
                "p25": None,
                "p75": None,
                "p95": None,
            }
        return {
            "count": int(len(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    return {
        "all_scores": desc(y_score),
        "positive_scores": desc(pos_scores),
        "negative_scores": desc(neg_scores),
        "neg_above_01": float(np.mean(neg_scores >= 0.10)) if len(neg_scores) > 0 else None,
        "neg_above_03": float(np.mean(neg_scores >= 0.30)) if len(neg_scores) > 0 else None,
        "neg_above_05": float(np.mean(neg_scores >= 0.50)) if len(neg_scores) > 0 else None,
        "neg_above_07": float(np.mean(neg_scores >= 0.70)) if len(neg_scores) > 0 else None,
        "pos_above_01": float(np.mean(pos_scores >= 0.10)) if len(pos_scores) > 0 else None,
        "pos_above_03": float(np.mean(pos_scores >= 0.30)) if len(pos_scores) > 0 else None,
        "pos_above_05": float(np.mean(pos_scores >= 0.50)) if len(pos_scores) > 0 else None,
        "pos_above_07": float(np.mean(pos_scores >= 0.70)) if len(pos_scores) > 0 else None,
    }


def sweep_image_thresholds(
    y_true,
    y_score,
    min_positive_boxes_arr,
    threshold_grid,
):
    rows = []

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    min_positive_boxes_arr = np.asarray(min_positive_boxes_arr).astype(int)

    for mpb in sorted(np.unique(min_positive_boxes_arr)):
        box_mask = (min_positive_boxes_arr >= mpb).astype(int)

        for thr in threshold_grid:
            y_pred = ((y_score >= thr).astype(int) * box_mask).astype(int)
            metrics = compute_medical_metrics(y_true, y_score, y_pred)

            rows.append({
                "min_positive_boxes": int(mpb),
                "image_score_thresh": float(thr),
                **metrics,
            })

    df = pd.DataFrame(rows)

    def pick_best(df_in, metric_name, maximize=True):
        if len(df_in) == 0:
            return None
        idx = df_in[metric_name].astype(float).idxmax() if maximize else df_in[metric_name].astype(float).idxmin()
        return df_in.loc[idx].to_dict()

    best = {
        "best_by_f1": pick_best(df, "f1_score", True),
        "best_by_balanced_accuracy": pick_best(df, "balanced_accuracy", True),
        "best_by_youden_j": pick_best(df, "youden_j", True),
        "best_by_specificity": pick_best(df, "specificity", True),
    }

    return df, best


# =========================================================
# INFERENCE
# =========================================================
@torch.no_grad()
def forward_model(model, images):
    try:
        return model(images)
    except TypeError:
        return model(images, None)


@torch.no_grad()
def run_evaluation(args):
    ensure_dir(args.output_dir)

    dataset = RSNADetectionEvalDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        image_size=args.image_size,
        apply_clahe=args.apply_clahe,
        norm_mode=args.norm_mode,
        to_3channel=args.to_3channel,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=(args.clahe_tile_grid_size, args.clahe_tile_grid_size),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Değişken Tanımı (KRİTİK): İlerleme çubuğu için toplam batch sayısı
    num_batches = len(loader)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = build_model_from_module(args)
    model = load_checkpoint_weights(model, args.checkpoint, device=device)
    model.to(device)
    model.eval()

    per_image_results = []
    image_rows = []

    print(f"Starting Evaluation on {len(dataset)} images...")

    for step, (images, metas) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        outputs = forward_model(model, images)
        
        # Temiz bir ekran için debug=False yapmanı öneririm
        decoded = decode_model_outputs(
            outputs=outputs,
            batch_size=images.size(0),
            image_size=args.image_size,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
            debug=False, # Üst üste yazı binmemesi için False yaptık
        )

        for meta, pred in zip(metas, decoded):
            orig_w = int(meta["orig_w"])
            orig_h = int(meta["orig_h"])

            pred_boxes_scaled = map_boxes_from_padded_to_original(
                boxes_xyxy=pred["boxes"],
                orig_w=orig_w,
                orig_h=orig_h,
                image_size=args.image_size,
            )
            pred_boxes_scaled = [clip_box_xyxy(b, orig_w, orig_h) for b in pred_boxes_scaled]
            gt_boxes = [clip_box_xyxy(b, orig_w, orig_h) for b in meta["gt_boxes"]]

            img_out = compute_image_level_outputs_for_record(
                pred_boxes=pred_boxes_scaled,
                pred_scores=pred["scores"],
                pred_labels=pred["labels"],
                agg_mode=args.image_agg_mode,
                topk=args.image_topk,
                min_box_score=args.image_box_score_thresh,
                min_box_area=args.image_min_box_area,
                min_positive_boxes=args.image_min_positive_boxes,
                image_score_thresh=args.image_score_thresh,
                max_boxes_for_image_score=args.image_max_boxes,
            )

            rec = {
                "patient_id": meta["patient_id"],
                "image_path": meta["image_path"],
                "label": int(meta["label"]),
                "image_score": float(img_out["image_score"]),
                "image_pred": int(img_out["image_pred"]),
                "num_valid_boxes_for_image": int(img_out["num_valid_boxes_for_image"]),
                "max_valid_score": float(img_out["max_valid_score"]),
                "mean_valid_score": float(img_out["mean_valid_score"]),
                "sum_valid_score": float(img_out["sum_valid_score"]),
                "max_valid_area": float(img_out["max_valid_area"]),
                "pred_boxes": pred_boxes_scaled,
                "pred_scores": [float(s) for s in pred["scores"]],
                "pred_labels": [int(l) for l in pred["labels"]],
                "gt_boxes": gt_boxes,
            }
            per_image_results.append(rec)

            image_rows.append({
                "patient_id": meta["patient_id"],
                "image_path": meta["image_path"],
                "gt_label": int(meta["label"]),
                "pred_label": int(img_out["image_pred"]),
                "image_score": float(img_out["image_score"]),
                "num_gt_boxes": len(gt_boxes),
                "num_pred_boxes_total": len(pred_boxes_scaled),
                "num_valid_boxes_for_image": int(img_out["num_valid_boxes_for_image"]),
                "max_valid_score": float(img_out["max_valid_score"]),
                "mean_valid_score": float(img_out["mean_valid_score"]),
                "sum_valid_score": float(img_out["sum_valid_score"]),
                "max_valid_area": float(img_out["max_valid_area"]),
            })

        # Tek satırda güncelleme
        sys.stdout.write(f"\r[EVAL] Progress: {step+1}/{num_batches} batches processed...")
        sys.stdout.flush()

    print("\n\nCalculation summary metrics...") # Döngü bitince yeni satıra geç

    # ... (Geri kalan metrik hesaplama ve kaydetme kodları aynı kalıyor)

    detection_metrics = compute_detection_summary(
        per_image_results=per_image_results,
        iou_thresholds=np.arange(0.50, 0.96, 0.05),
        score_threshold_for_summary=args.summary_score_thresh,
    )

    y_true = [r["label"] for r in per_image_results]
    y_score = [r["image_score"] for r in per_image_results]
    y_pred = [r["image_pred"] for r in per_image_results]

    medical_metrics = compute_medical_metrics(y_true, y_score, y_pred)
    score_distribution = summarize_score_distribution(y_true, y_score)

    image_df = pd.DataFrame(image_rows)
    image_csv_path = os.path.join(args.output_dir, "image_level_predictions.csv")
    image_df.to_csv(image_csv_path, index=False)

    with open(os.path.join(args.output_dir, "detailed_detection_results.json"), "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, indent=2, ensure_ascii=False)

    curves = {}
    if len(np.unique(y_true)) > 1:
        fpr, tpr, roc_thr = roc_curve(y_true, y_score)
        prec_curve, rec_curve, pr_thr = precision_recall_curve(y_true, y_score)

        curves["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thr.tolist(),
        }
        curves["pr_curve"] = {
            "precision": prec_curve.tolist(),
            "recall": rec_curve.tolist(),
            "thresholds": pr_thr.tolist(),
        }

        with open(os.path.join(args.output_dir, "curves.json"), "w", encoding="utf-8") as f:
            json.dump(curves, f, indent=2, ensure_ascii=False)

    threshold_sweep_report = None
    if args.enable_threshold_sweep:
        threshold_grid = np.linspace(
            args.sweep_min_thresh,
            args.sweep_max_thresh,
            args.sweep_num_points
        )

        min_positive_boxes_arr = [max(1, r["num_valid_boxes_for_image"]) for r in per_image_results]
        sweep_df, best_thresholds = sweep_image_thresholds(
            y_true=y_true,
            y_score=y_score,
            min_positive_boxes_arr=min_positive_boxes_arr,
            threshold_grid=threshold_grid,
        )
        sweep_csv_path = os.path.join(args.output_dir, "image_threshold_sweep.csv")
        sweep_df.to_csv(sweep_csv_path, index=False)

        threshold_sweep_report = {
            "sweep_csv_path": sweep_csv_path,
            "best_thresholds": best_thresholds,
        }

    final_report = {
        "config": vars(args),
        "num_images": len(per_image_results),
        "detection_metrics": detection_metrics,
        "medical_image_level_metrics": medical_metrics,
        "image_score_distribution": score_distribution,
        "threshold_sweep": threshold_sweep_report,
    }

    with open(os.path.join(args.output_dir, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("RSNA DETECTION EVALUATION SUMMARY")
    print("=" * 100)

    print("\n[Detection Metrics]")
    print(f"AP50                            : {detection_metrics['AP50']:.6f}")
    print(f"AP75                            : {detection_metrics['AP75']:.6f}")
    print(f"mAP@[0.50:0.95]                 : {detection_metrics['mAP50_95']:.6f}")
    print(f"Precision @ summary threshold   : {detection_metrics['detection_precision_at_score_thresh']:.6f}")
    print(f"Recall @ summary threshold      : {detection_metrics['detection_recall_at_score_thresh']:.6f}")
    print(f"F1 @ summary threshold          : {detection_metrics['detection_f1_at_score_thresh']:.6f}")
    print(f"Mean matched IoU                : {detection_metrics['mean_matched_iou']:.6f}")
    print(f"Median matched IoU              : {detection_metrics['median_matched_iou']:.6f}")
    print(f"TP / FP / FN                    : {detection_metrics['tp']} / {detection_metrics['fp']} / {detection_metrics['fn']}")

    print("\n[Medical Image-Level Metrics]")
    print(f"Accuracy                        : {medical_metrics['accuracy']:.6f}")
    print(f"Sensitivity / Recall            : {medical_metrics['sensitivity_recall']:.6f}")
    print(f"Specificity                     : {medical_metrics['specificity']:.6f}")
    print(f"Precision / PPV                 : {medical_metrics['precision_ppv']:.6f}")
    print(f"NPV                             : {medical_metrics['npv']:.6f}")
    print(f"F1-score                        : {medical_metrics['f1_score']:.6f}")
    print(f"Balanced Accuracy               : {medical_metrics['balanced_accuracy']:.6f}")
    print(f"Youden J                        : {medical_metrics['youden_j']:.6f}")
    print(f"ROC-AUC                         : {medical_metrics['roc_auc']}")
    print(f"PR-AUC                          : {medical_metrics['pr_auc']}")
    print(f"TP / TN / FP / FN               : {medical_metrics['tp']} / {medical_metrics['tn']} / {medical_metrics['fp']} / {medical_metrics['fn']}")

    print("\n[Image Score Distribution]")
    print(f"All score mean                  : {score_distribution['all_scores']['mean']}")
    print(f"Positive score mean             : {score_distribution['positive_scores']['mean']}")
    print(f"Negative score mean             : {score_distribution['negative_scores']['mean']}")
    print(f"Negatives >= 0.30               : {score_distribution['neg_above_03']}")
    print(f"Negatives >= 0.50               : {score_distribution['neg_above_05']}")
    print(f"Negatives >= 0.70               : {score_distribution['neg_above_07']}")

    if threshold_sweep_report is not None:
        print("\n[Threshold Sweep]")
        best_f1 = threshold_sweep_report["best_thresholds"]["best_by_f1"]
        best_bacc = threshold_sweep_report["best_thresholds"]["best_by_balanced_accuracy"]
        if best_f1 is not None:
            print(
                f"Best by F1                      : "
                f"thr={best_f1['image_score_thresh']:.4f}, "
                f"min_boxes={best_f1['min_positive_boxes']}, "
                f"F1={best_f1['f1_score']:.6f}, "
                f"Spec={best_f1['specificity']:.6f}, "
                f"Sens={best_f1['sensitivity_recall']:.6f}"
            )
        if best_bacc is not None:
            print(
                f"Best by Balanced Accuracy       : "
                f"thr={best_bacc['image_score_thresh']:.4f}, "
                f"min_boxes={best_bacc['min_positive_boxes']}, "
                f"BAcc={best_bacc['balanced_accuracy']:.6f}, "
                f"Spec={best_bacc['specificity']:.6f}, "
                f"Sens={best_bacc['sensitivity_recall']:.6f}"
            )

    print(f"\nSaved to: {args.output_dir}")
    print("=" * 100)


# =========================================================
# ARGS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Only needed if CSV does not contain image_path.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # preprocessing
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--apply_clahe", action="store_true")
    parser.add_argument("--norm_mode", type=str, default="imagenet", choices=["imagenet", "minmax_01"])
    parser.add_argument("--to_3channel", action="store_true")
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--clahe_tile_grid_size", type=int, default=8)

    # model import/build
    parser.add_argument("--model_module", type=str, default="model_diffusion_guided_deformable_detr")
    parser.add_argument("--model_builder", type=str, default="")

    # architecture
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_feature_levels", type=int, default=4)
    parser.add_argument("--backbone_name", type=str, default="swin_tiny_patch4_window7_224")
    parser.add_argument("--fusion_mode", type=str, default="hybrid")
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_points", type=int, default=4)

    # eval runtime
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda")

    # detection thresholds
    parser.add_argument("--score_thresh", type=float, default=0.01,
                        help="Min detection box score to keep before NMS.")
    parser.add_argument("--nms_thresh", type=float, default=0.50)
    parser.add_argument("--summary_score_thresh", type=float, default=0.30,
                        help="Threshold for TP/FP/FN summary at IoU=0.50.")

    # improved image-level logic
    parser.add_argument("--image_agg_mode", type=str, default="topk_mean",
                        choices=["max", "topk_mean", "topk_sum", "logsumexp", "area_weighted", "score_area_hybrid"])
    parser.add_argument("--image_topk", type=int, default=3)
    parser.add_argument("--image_box_score_thresh", type=float, default=0.20)
    parser.add_argument("--image_min_box_area", type=float, default=2048.0)
    parser.add_argument("--image_min_positive_boxes", type=int, default=2)
    parser.add_argument("--image_score_thresh", type=float, default=0.35)
    parser.add_argument("--image_max_boxes", type=int, default=20)

    # threshold sweep
    parser.add_argument("--enable_threshold_sweep", action="store_true")
    parser.add_argument("--sweep_min_thresh", type=float, default=0.05)
    parser.add_argument("--sweep_max_thresh", type=float, default=0.95)
    parser.add_argument("--sweep_num_points", type=int, default=91)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)