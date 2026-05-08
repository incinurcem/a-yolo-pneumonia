# Wrapper entrypoint for single-image inference
#s
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tek görüntü üzerinde inference yapar:
- classifier probability
- opsiyonel GAN anomaly map
- opsiyonel Grad-CAM
- JSON özet + görseller

Örnek:
python scripts/infer/predict_single.py \
    --image data/images_png/0004cfab-14fd-4e49-80ba-63a80b6bddd6.png \
    --classifier-checkpoint outputs/classifier/best.pt \
    --gan-checkpoint outputs/gan/best_generator.pt \
    --output-dir outputs/infer/single_case \
    --save-gradcam
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pneumonia probability for a single image.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--gan-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save-gradcam", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class PneumoniaClassifier(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=False, in_chans=3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=1, in_chans=in_chans)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
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
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.b = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.b(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.target_layer = self._find_last_conv(model)
        self.h1 = self.target_layer.register_forward_hook(self._forward_hook)
        self.h2 = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _find_last_conv(self, model):
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("GradCAM için Conv2d bulunamadı.")
        return last_conv

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        logit.sum().backward(retain_graph=True)
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
        self.h1.remove()
        self.h2.remove()


def load_classifier(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PneumoniaClassifier(
        model_name=ckpt.get("model_name", "resnet50"),
        pretrained=False,
        in_chans=int(ckpt.get("in_chans", 3)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, int(ckpt.get("image_size", 512)), ckpt.get("mean", [0.485, 0.456, 0.406]), ckpt.get("std", [0.229, 0.224, 0.225]), float(ckpt.get("threshold", 0.5))


def load_gan(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
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


def preprocess_for_classifier(img_path: str, image_size: int, mean, std):
    raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {img_path}")
    raw = cv2.resize(raw, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = np.stack([raw, raw, raw], axis=-1).astype(np.float32) / 255.0
    norm = (rgb - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = np.transpose(norm, (2, 0, 1))
    return raw, torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)


def preprocess_for_gan(raw_uint8: np.ndarray):
    arr = raw_uint8.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    arr = np.expand_dims(arr, 0)
    return torch.tensor(arr, dtype=torch.float32)


def save_overlay(raw_uint8: np.ndarray, heatmap: np.ndarray, out_path: Path):
    hm = (heatmap * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    gray = cv2.cvtColor(raw_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray, 0.6, hm, 0.4, 0)
    cv2.imwrite(str(out_path), overlay)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cls_model, image_size, mean, std, ckpt_thr = load_classifier(args.classifier_checkpoint, args.device)
    gan_model = load_gan(args.gan_checkpoint, args.device) if args.gan_checkpoint else None

    threshold = args.threshold if args.threshold is not None else ckpt_thr

    raw_uint8, x_cls = preprocess_for_classifier(args.image, image_size, mean, std)
    x_cls = x_cls.to(args.device)

    with torch.no_grad():
        prob = float(torch.sigmoid(cls_model(x_cls)).item())

    pred = int(prob >= threshold)

    result = {
        "image": args.image,
        "probability": prob,
        "threshold": threshold,
        "prediction": pred,
        "prediction_label": "pneumonia" if pred == 1 else "normal",
    }

    if gan_model is not None:
        x_gan = preprocess_for_gan(raw_uint8).to(args.device)
        with torch.no_grad():
            recon = gan_model(x_gan)
            anomaly = torch.abs(x_gan - recon).squeeze().cpu().numpy()

        anomaly = anomaly - anomaly.min()
        if anomaly.max() > 0:
            anomaly = anomaly / anomaly.max()

        result["gan_score"] = float(anomaly.mean())
        cv2.imwrite(str(output_dir / "anomaly_map.png"), (anomaly * 255).astype(np.uint8))
        save_overlay(raw_uint8, anomaly, output_dir / "anomaly_overlay.png")

    if args.save_gradcam:
        gradcam = GradCAM(cls_model)
        cam = gradcam(x_cls)
        gradcam.close()

        result["cam_score"] = float(cam.mean())
        cv2.imwrite(str(output_dir / "gradcam_map.png"), (cam * 255).astype(np.uint8))
        save_overlay(raw_uint8, cam, output_dir / "gradcam_overlay.png")

    with open(output_dir / "prediction.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[OK] Tek görüntü inference tamamlandı.")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()