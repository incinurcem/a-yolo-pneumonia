# Wrapper entrypoint for batch inference
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#s
"""
Batch inference:
- Classifier olasılığı üretir
- Opsiyonel GAN reconstruction + anomaly map üretir
- Opsiyonel Grad-CAM üretir
- CSV + görsel çıktı verir

Örnek:
python scripts/infer/predict_batch.py \
    --manifest data/splits/test_classifier.csv \
    --classifier-checkpoint outputs/classifier/best.pt \
    --output-dir outputs/infer/test_batch \
    --gan-checkpoint outputs/gan/best_generator.pt \
    --save-anomaly-maps \
    --save-gradcam
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for classifier + GAN + GradCAM.")
    parser.add_argument("--manifest", type=str, default=None, help="CSV manifest")
    parser.add_argument("--input-dir", type=str, default=None, help="Alternatif: doğrudan görüntü klasörü")
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--gan-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save-anomaly-maps", action="store_true")
    parser.add_argument("--save-gradcam", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class InferenceDataset(Dataset):
    def __init__(self, manifest: Optional[str], input_dir: Optional[str], image_size: int, mean=None, std=None):
        if manifest is None and input_dir is None:
            raise ValueError("manifest veya input-dir verilmelidir.")

        self.image_size = image_size
        self.mean = np.array(mean if mean is not None else [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array(std if std is not None else [0.229, 0.224, 0.225], dtype=np.float32)

        if manifest is not None:
            df = pd.read_csv(manifest)
            self.items = []
            for _, row in df.iterrows():
                pid = row["patient_id"] if "patient_id" in row else Path(row["image_path"]).stem
                self.items.append(
                    {
                        "patient_id": pid,
                        "img_path": row["image_path"],
                        "label": int(row["label"]) if "label" in row else -1,
                    }
                )
        else:
            root = Path(input_dir)
            self.items = []
            for p in sorted(root.glob("*")):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    self.items.append(
                        {
                            "patient_id": p.stem,
                            "img_path": str(p),
                            "label": -1,
                        }
                    )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = cv2.imread(item["img_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {item['img_path']}")

        raw = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        rgb = np.stack([raw, raw, raw], axis=-1).astype(np.float32) / 255.0
        norm = (rgb - self.mean) / self.std
        tensor = np.transpose(norm, (2, 0, 1))

        gan_tensor = raw.astype(np.float32) / 255.0
        gan_tensor = np.expand_dims(gan_tensor, 0)

        return {
            "patient_id": item["patient_id"],
            "img_path": item["img_path"],
            "label": item["label"],
            "image_cls": torch.tensor(tensor, dtype=torch.float32),
            "image_gan": torch.tensor(gan_tensor, dtype=torch.float32),
            "raw_uint8": raw,
        }


class PneumoniaClassifier(nn.Module):
    def __init__(self, model_name: str = "resnet50", pretrained: bool = False, in_chans: int = 3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=1, in_chans=in_chans)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class GeneratorUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = torch.sigmoid(self.out_conv(d1))
        return out


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits.sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Conv2d katmanı bulunamadı, GradCAM uygulanamadı.")
    return last_conv


def load_classifier(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "resnet50")
    image_size = int(ckpt.get("image_size", 512))
    in_chans = int(ckpt.get("in_chans", 3))
    mean = ckpt.get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std", [0.229, 0.224, 0.225])
    threshold = float(ckpt.get("threshold", 0.5))

    model = PneumoniaClassifier(model_name=model_name, pretrained=False, in_chans=in_chans)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return model, image_size, mean, std, threshold


def load_gan(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = GeneratorUNet(
        in_ch=int(ckpt.get("in_chans", 1)),
        out_ch=int(ckpt.get("out_chans", 1)),
        base=int(ckpt.get("base_channels", 32)),
    )
    state = ckpt.get("generator_state_dict", ckpt.get("state_dict"))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_overlay(gray_uint8: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    gray_rgb = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray_rgb, 0.6, heatmap_color, 0.4, 0)
    return overlay


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_dir = output_dir / "anomaly_maps"
    gradcam_dir = output_dir / "gradcam_maps"
    overlay_dir = output_dir / "overlays"

    if args.save_anomaly_maps:
        anomaly_dir.mkdir(parents=True, exist_ok=True)
    if args.save_gradcam:
        gradcam_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    cls_model, image_size, mean, std, threshold = load_classifier(args.classifier_checkpoint, args.device)
    gan_model = load_gan(args.gan_checkpoint, args.device) if args.gan_checkpoint else None

    dataset = InferenceDataset(args.manifest, args.input_dir, image_size=image_size, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    gradcam = None
    if args.save_gradcam:
        gradcam = GradCAM(cls_model, get_last_conv_layer(cls_model))

    used_threshold = args.threshold if args.threshold is not None else threshold

    rows = []
    for batch in tqdm(loader, desc="Batch inference"):
        x_cls = batch["image_cls"].to(args.device)
        x_gan = batch["image_gan"].to(args.device)

        with torch.no_grad():
            logits = cls_model(x_cls)
            probs = torch.sigmoid(logits).cpu().numpy()

        recon = None
        anomaly_maps = None
        if gan_model is not None:
            with torch.no_grad():
                recon = gan_model(x_gan)
                anomaly_maps = torch.abs(x_gan - recon).cpu().numpy()

        bsz = x_cls.size(0)
        for i in range(bsz):
            patient_id = batch["patient_id"][i]
            img_path = batch["img_path"][i]
            label = int(batch["label"][i])
            prob = float(probs[i])
            pred = int(prob >= used_threshold)

            raw_uint8 = batch["raw_uint8"][i].numpy() if hasattr(batch["raw_uint8"][i], "numpy") else batch["raw_uint8"][i]

            gan_score = None
            if anomaly_maps is not None:
                amap = anomaly_maps[i, 0]
                amap = amap - amap.min()
                if amap.max() > 0:
                    amap = amap / amap.max()
                gan_score = float(amap.mean())

                if args.save_anomaly_maps:
                    cv2.imwrite(str(anomaly_dir / f"{patient_id}.png"), (amap * 255).astype(np.uint8))

                if args.save_overlays:
                    overlay = make_overlay(raw_uint8, amap)
                    cv2.imwrite(str(overlay_dir / f"{patient_id}_anomaly_overlay.png"), overlay)

            cam_score = None
            if gradcam is not None:
                cam = gradcam(x_cls[i:i + 1])
                cam_score = float(cam.mean())
                if args.save_gradcam:
                    cv2.imwrite(str(gradcam_dir / f"{patient_id}.png"), (cam * 255).astype(np.uint8))
                if args.save_overlays:
                    overlay = make_overlay(raw_uint8, cam)
                    cv2.imwrite(str(overlay_dir / f"{patient_id}_gradcam_overlay.png"), overlay)

            rows.append(
                {
                    "patient_id": patient_id,
                    "img_path": img_path,
                    "label": label,
                    "probability": prob,
                    "prediction": pred,
                    "threshold": used_threshold,
                    "gan_score": gan_score,
                    "cam_score": cam_score,
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "batch_predictions.csv", index=False)

    if gradcam is not None:
        gradcam.close()

    summary = {
        "num_samples": len(dataset),
        "output_dir": str(output_dir),
        "used_threshold": float(used_threshold),
        "saved_anomaly_maps": bool(args.save_anomaly_maps),
        "saved_gradcam": bool(args.save_gradcam),
        "saved_overlays": bool(args.save_overlays),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Batch inference tamamlandı.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()