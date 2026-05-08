# Wrapper entrypoint for FastAPI app
#s
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI tabanlı inference servisi.

Endpoint'ler:
- GET  /health
- POST /predict-file

predict-file çıktısı:
- pneumonia probability
- predicted label
- opsiyonel GAN anomaly score
- opsiyonel Grad-CAM / anomaly overlay base64

Örnek:
python scripts/serve/launch_api.py \
    --classifier-checkpoint outputs/classifier/best.pt \
    --gan-checkpoint outputs/gan/best_generator.pt \
    --host 0.0.0.0 \
    --port 8000
"""

import argparse
import base64
import io
from typing import Optional

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Launch FastAPI for pneumonia demo.")
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--gan-checkpoint", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
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
    def __init__(self, model):
        self.model = model
        self.target_layer = self._find_last_conv(model)
        self.activations = None
        self.gradients = None
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

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
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
        prob = float(torch.sigmoid(logit).item())
        return cam.astype(np.float32), prob


def np_to_base64_png(arr_uint8: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", arr_uint8)
    if not success:
        raise RuntimeError("PNG encode başarısız.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def overlay_heatmap(gray_uint8: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heat_uint8 = (heatmap * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    gray_bgr = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray_bgr, 0.6, heat_color, 0.4, 0)
    return overlay


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


def pil_to_gray_uint8(data: bytes, image_size: int) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L")
    arr = np.array(img)
    arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return arr.astype(np.uint8)


def preprocess_classifier(gray_uint8: np.ndarray, mean, std):
    rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1).astype(np.float32) / 255.0
    norm = (rgb - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = np.transpose(norm, (2, 0, 1))
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)


def preprocess_gan(gray_uint8: np.ndarray):
    arr = gray_uint8.astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    arr = np.expand_dims(arr, 0)
    return torch.tensor(arr, dtype=torch.float32)


args = parse_args()
app = FastAPI(title="Pneumonia Classifier + GAN API")

DEVICE = args.device
CLS_MODEL, IMAGE_SIZE, MEAN, STD, THRESHOLD = load_classifier(args.classifier_checkpoint, DEVICE)
GAN_MODEL = load_gan(args.gan_checkpoint, DEVICE) if args.gan_checkpoint else None
CAM = GradCAM(CLS_MODEL)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "image_size": IMAGE_SIZE,
        "gan_enabled": GAN_MODEL is not None,
    }


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    gray_uint8 = pil_to_gray_uint8(content, IMAGE_SIZE)

    x_cls = preprocess_classifier(gray_uint8, MEAN, STD).to(DEVICE)

    with torch.no_grad():
        logit = CLS_MODEL(x_cls)
        prob = float(torch.sigmoid(logit).item())

    pred = int(prob >= THRESHOLD)

    response = {
        "filename": file.filename,
        "probability": prob,
        "threshold": THRESHOLD,
        "prediction": pred,
        "prediction_label": "pneumonia" if pred == 1 else "normal",
        "original_image_b64": np_to_base64_png(gray_uint8),
    }

    cam_map, _ = CAM(x_cls)
    cam_overlay = overlay_heatmap(gray_uint8, cam_map)
    response["gradcam_score"] = float(cam_map.mean())
    response["gradcam_map_b64"] = np_to_base64_png((cam_map * 255).astype(np.uint8))
    response["gradcam_overlay_b64"] = np_to_base64_png(cam_overlay)

    if GAN_MODEL is not None:
        x_gan = preprocess_gan(gray_uint8).to(DEVICE)
        with torch.no_grad():
            recon = GAN_MODEL(x_gan)
            anomaly = torch.abs(x_gan - recon).squeeze().cpu().numpy()

        anomaly = anomaly - anomaly.min()
        if anomaly.max() > 0:
            anomaly = anomaly / anomaly.max()

        anomaly_overlay = overlay_heatmap(gray_uint8, anomaly)

        response["gan_score"] = float(anomaly.mean())
        response["anomaly_map_b64"] = np_to_base64_png((anomaly * 255).astype(np.uint8))
        response["anomaly_overlay_b64"] = np_to_base64_png(anomaly_overlay)

    return response


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)