# -*- coding: utf-8 -*-
"""
predict_rsna_detection.py

Project-aligned prediction script for:
Diffusion-Guided Deformable DETR on RSNA pneumonia detection

Supports:
- single image
- image folder
- CSV with image_path
- CSV with patientId + image_dir
- master CSV from preprocessing pipeline

Aligned with train/eval pipeline:
- grayscale PNG reading
- optional CLAHE
- optional 3-channel conversion
- LongestMaxSize + PadIfNeeded
- normalization by config
- correct reverse mapping from padded square image to original image

Saves:
- predictions.csv
- predictions.json
- visualizations/*.png (optional)
"""

import os
import json
import glob
import argparse
import importlib
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


# =========================================================
# ENV
# =========================================================
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# =========================================================
# BASIC UTILS
# =========================================================

def constant_pad_kwargs():
    return {
        "border_mode": cv2.BORDER_CONSTANT,
        "value": 0,
        "mask_value": 0,
    }

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def infer_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(max(w - 1, 0))))
    y1 = max(0.0, min(float(y1), float(max(h - 1, 0))))
    x2 = max(0.0, min(float(x2), float(max(w, 0))))
    y2 = max(0.0, min(float(y2), float(max(h, 0))))
    return [x1, y1, x2, y2]


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


def cxcywh_to_xyxy_tensor(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


# =========================================================
# PREPROCESSING (MATCH TRAIN/EVAL)
# =========================================================
def read_png_grayscale(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

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


def get_predict_transform(image_size: int, norm_mode: str, to_3channel: bool):
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


def compute_resize_pad_params(orig_h: int, orig_w: int, image_size: int):
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


def map_boxes_from_padded_to_original(boxes_xyxy, orig_w, orig_h, image_size):
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
# INPUT COLLECTION
# =========================================================
def resolve_csv_image_paths(csv_path, image_dir=None):
    df = pd.read_csv(csv_path)

    image_path_col = infer_column(df, ["image_path", "image_file", "filepath", "path", "file_path"], required=False)
    patient_col = infer_column(df, ["patientId", "patient_id", "image_id", "id"], required=False)

    paths = []
    if image_path_col is not None:
        for p in df[image_path_col].astype(str).tolist():
            if os.path.exists(p):
                paths.append(p)
            elif image_dir is not None:
                maybe = os.path.join(image_dir, os.path.basename(p))
                paths.append(maybe)
            else:
                paths.append(p)
        return sorted(list(dict.fromkeys(paths)))

    if patient_col is not None:
        if image_dir is None:
            raise ValueError(
                "CSV has no image_path column. Provide --image_dir so images can be resolved as patientId.png"
            )

        for pid in df[patient_col].astype(str).tolist():
            candidates = [
                os.path.join(image_dir, f"{pid}.png"),
                os.path.join(image_dir, f"{pid}.jpg"),
                os.path.join(image_dir, f"{pid}.jpeg"),
            ]
            found = None
            for c in candidates:
                if os.path.exists(c):
                    found = c
                    break
            if found is None:
                found = candidates[0]
            paths.append(found)

        return sorted(list(dict.fromkeys(paths)))

    raise ValueError("CSV must contain either image_path-like column or patientId-like column.")


def collect_image_paths(input_path, image_dir=None):
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()

        if ext == ".csv":
            return resolve_csv_image_paths(input_path, image_dir=image_dir)

        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            return [input_path]

        raise ValueError(f"Unsupported input file type: {ext}")

    if os.path.isdir(input_path):
        image_exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        paths = []
        for e in image_exts:
            paths.extend(glob.glob(os.path.join(input_path, e)))
        return sorted(paths)

    raise FileNotFoundError(f"Input not found: {input_path}")


# =========================================================
# DATASET
# =========================================================
class PredictDataset(Dataset):
    def __init__(
        self,
        image_paths,
        image_size=384,
        apply_clahe=False,
        norm_mode="imagenet",
        to_3channel=False,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
    ):
        self.image_paths = image_paths
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.norm_mode = norm_mode
        self.to_3channel = to_3channel
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.transform = get_predict_transform(
            image_size=image_size,
            norm_mode=norm_mode,
            to_3channel=to_3channel,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_png_grayscale(image_path)
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

        return {
            "image": image,
            "image_path": image_path,
            "orig_h": orig_h,
            "orig_w": orig_w,
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    metas = batch
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
        return cls(
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

    raise AttributeError(
        f"Could not build model from module='{args.model_module}'."
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
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
        first = outputs[0]
        if isinstance(first, (dict, list)):
            return first
    return outputs


@torch.no_grad()
def decode_model_outputs(outputs, batch_size, image_size, score_thresh=0.25, nms_thresh=0.5):
    outputs = unwrap_outputs(outputs)
    decoded = []

    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
        for out in outputs:
            boxes = out.get("boxes", torch.empty((0, 4)))
            scores = out.get("scores", torch.empty((0,)))
            labels = out.get("labels", torch.empty((0,), dtype=torch.long))

            boxes = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
            scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
            labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

            keep = scores >= score_thresh
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if len(boxes) > 0:
                keep_nms = nms_numpy(boxes, scores, iou_thresh=nms_thresh)
                boxes = boxes[keep_nms]
                scores = scores[keep_nms]
                labels = labels[keep_nms]

            decoded.append({
                "boxes": boxes.tolist(),
                "scores": scores.tolist(),
                "labels": labels.tolist(),
            })
        return decoded

    if isinstance(outputs, dict) and "pred_logits" in outputs and "pred_boxes" in outputs:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        B, Q, C = pred_logits.shape
        if B != batch_size:
            raise ValueError(f"Batch mismatch: outputs batch={B}, expected={batch_size}")

        for b in range(B):
            logits_b = pred_logits[b]
            boxes_b = pred_boxes[b]

            if C >= 2:
                probs = logits_b.softmax(dim=-1)
                scores, labels = probs[:, :-1].max(dim=-1)
                labels = labels + 1
            else:
                probs = logits_b.sigmoid()
                scores, labels = probs.max(dim=-1)
                labels = labels + 1

            boxes_xyxy = cxcywh_to_xyxy_tensor(boxes_b)

            if float(boxes_xyxy.max()) <= 1.5:
                boxes_xyxy[:, [0, 2]] *= image_size
                boxes_xyxy[:, [1, 3]] *= image_size

            boxes_xyxy = boxes_xyxy.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            keep = scores >= score_thresh
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]
            labels = labels[keep]

            clipped = [clip_box_xyxy(b.tolist(), image_size, image_size) for b in boxes_xyxy]
            boxes_xyxy = np.asarray(clipped, dtype=np.float32) if len(clipped) > 0 else np.zeros((0, 4), dtype=np.float32)

            if len(boxes_xyxy) > 0:
                keep_nms = nms_numpy(boxes_xyxy, scores, iou_thresh=nms_thresh)
                boxes_xyxy = boxes_xyxy[keep_nms]
                scores = scores[keep_nms]
                labels = labels[keep_nms]

            decoded.append({
                "boxes": boxes_xyxy.tolist(),
                "scores": scores.tolist(),
                "labels": labels.tolist(),
            })
        return decoded

    if isinstance(outputs, dict) and "boxes" in outputs and "scores" in outputs:
        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs.get("labels", None)

        if isinstance(boxes, list) and len(boxes) == batch_size:
            for i in range(batch_size):
                bx = boxes[i]
                sc = scores[i]
                lb = labels[i] if labels is not None else np.ones((len(sc),), dtype=np.int64)

                bx = bx.detach().cpu().numpy() if isinstance(bx, torch.Tensor) else np.asarray(bx)
                sc = sc.detach().cpu().numpy() if isinstance(sc, torch.Tensor) else np.asarray(sc)
                lb = lb.detach().cpu().numpy() if isinstance(lb, torch.Tensor) else np.asarray(lb)

                keep = sc >= score_thresh
                bx = bx[keep]
                sc = sc[keep]
                lb = lb[keep]

                if len(bx) > 0:
                    keep_nms = nms_numpy(bx, sc, iou_thresh=nms_thresh)
                    bx = bx[keep_nms]
                    sc = sc[keep_nms]
                    lb = lb[keep_nms]

                decoded.append({
                    "boxes": bx.tolist(),
                    "scores": sc.tolist(),
                    "labels": lb.tolist(),
                })
            return decoded

    raise ValueError("Unsupported model output format.")


# =========================================================
# VISUALIZATION
# =========================================================
def draw_predictions(image_gray, boxes, scores, labels, image_score=None):
    if image_gray.ndim == 2:
        canvas = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image_gray.copy()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"cls={label} score={score:.3f}"
        cv2.putText(
            canvas,
            text,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if image_score is not None:
        cv2.putText(
            canvas,
            f"image_score={image_score:.3f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


# =========================================================
# MAIN PREDICTION
# =========================================================
@torch.no_grad()
def forward_model(model, images):
    try:
        return model(images)
    except TypeError:
        return model(images, None)


@torch.no_grad()
def run_prediction(args):
    ensure_dir(args.output_dir)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    ensure_dir(vis_dir)

    image_paths = collect_image_paths(args.input_path, image_dir=args.image_dir)
    print(f"Found {len(image_paths)} images.")

    dataset = PredictDataset(
        image_paths=image_paths,
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

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = build_model_from_module(args)
    model = load_checkpoint_weights(model, args.checkpoint, device=device)
    model.to(device)
    model.eval()

    all_json_records = []
    all_csv_rows = []

    for images, metas in loader:
        images = images.to(device, non_blocking=True)

        outputs = forward_model(model, images)
        decoded = decode_model_outputs(
            outputs=outputs,
            batch_size=images.size(0),
            image_size=args.image_size,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
        )

        for meta, pred in zip(metas, decoded):
            image_path = meta["image_path"]
            orig_h = meta["orig_h"]
            orig_w = meta["orig_w"]

            pred_boxes_scaled = map_boxes_from_padded_to_original(
                boxes_xyxy=pred["boxes"],
                orig_w=orig_w,
                orig_h=orig_h,
                image_size=args.image_size,
            )
            pred_boxes_scaled = [clip_box_xyxy(b, orig_w, orig_h) for b in pred_boxes_scaled]

            image_score = max(pred["scores"]) if len(pred["scores"]) > 0 else 0.0

            record = {
                "image_path": image_path,
                "image_score": float(image_score),
                "num_predictions": len(pred_boxes_scaled),
                "predictions": [],
            }

            for box, score, label in zip(pred_boxes_scaled, pred["scores"], pred["labels"]):
                x1, y1, x2, y2 = box
                row = {
                    "image_path": image_path,
                    "image_score": float(image_score),
                    "label": int(label),
                    "score": float(score),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                }
                all_csv_rows.append(row)
                record["predictions"].append(row)

            if len(pred_boxes_scaled) == 0:
                all_csv_rows.append({
                    "image_path": image_path,
                    "image_score": float(image_score),
                    "label": -1,
                    "score": 0.0,
                    "x1": np.nan,
                    "y1": np.nan,
                    "x2": np.nan,
                    "y2": np.nan,
                    "width": np.nan,
                    "height": np.nan,
                })

            all_json_records.append(record)

            if args.save_visualizations:
                image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image_gray is not None:
                    drawn = draw_predictions(
                        image_gray=image_gray,
                        boxes=pred_boxes_scaled,
                        scores=pred["scores"],
                        labels=pred["labels"],
                        image_score=image_score,
                    )
                    save_name = os.path.basename(image_path)
                    save_path = os.path.join(vis_dir, save_name)
                    cv2.imwrite(save_path, drawn)

    pred_csv_path = os.path.join(args.output_dir, "predictions.csv")
    pred_json_path = os.path.join(args.output_dir, "predictions.json")

    pd.DataFrame(all_csv_rows).to_csv(pred_csv_path, index=False)
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(all_json_records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("RSNA DETECTION PREDICTION FINISHED")
    print("=" * 100)
    print(f"Images processed      : {len(image_paths)}")
    print(f"Predictions CSV       : {pred_csv_path}")
    print(f"Predictions JSON      : {pred_json_path}")
    if args.save_visualizations:
        print(f"Visualizations folder : {vis_dir}")
    print("=" * 100)


# =========================================================
# ARGS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True,
                        help="Single image, image folder, or CSV")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Needed if input CSV has patientId but not image_path")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_module", type=str, default="model_diffusion_guided_deformable_detr")
    parser.add_argument("--model_builder", type=str, default="")

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

    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--apply_clahe", action="store_true")
    parser.add_argument("--norm_mode", type=str, default="imagenet", choices=["imagenet", "minmax_01"])
    parser.add_argument("--to_3channel", action="store_true")
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--clahe_tile_grid_size", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--score_thresh", type=float, default=0.01)
    parser.add_argument("--nms_thresh", type=float, default=0.50)

    parser.add_argument("--save_visualizations", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(args)