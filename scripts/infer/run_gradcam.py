# Wrapper entrypoint for Grad-CAM visualization
#s
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classifier checkpoint ile Grad-CAM üretir.

Kullanım:
- Tek görüntü
- Manifest CSV
- Klasör

Çıktılar:
- gradcam haritası
- overlay
- CSV özet

Örnek:
python scripts/infer/run_gradcam.py \
    --manifest data/splits/test_classifier.csv \
    --checkpoint outputs/classifier/best.pt \
    --output-dir outputs/gradcam/test
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for pneumonia classifier.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class PneumoniaClassifier(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=False, in_chans=3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=1, in_chans=in_chans)

    def forward(self, x):
        return self.backbone(x).squeeze(1)


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.target_layer = self._find_last_conv(model)
        self.activations = None
        self.gradients = None
        self.h1 = self.target_layer.register_forward_hook(self._forward_hook)
        self.h2 = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _find_last_conv(self, model: nn.Module):
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("Conv2d layer bulunamadı.")
        return last_conv

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        score = logit.sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32), float(torch.sigmoid(logit).item())

    def close(self):
        self.h1.remove()
        self.h2.remove()


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = PneumoniaClassifier(
        model_name=ckpt.get("model_name", "resnet50"),
        pretrained=False,
        in_chans=int(ckpt.get("in_chans", 3)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, int(ckpt.get("image_size", 512)), ckpt.get("mean", [0.485, 0.456, 0.406]), ckpt.get("std", [0.229, 0.224, 0.225])


def gather_items(args) -> List[Tuple[str, str]]:
    items = []
    if args.image is not None:
        p = Path(args.image)
        items.append((p.stem, str(p)))
        return items

    if args.manifest is not None:
        df = pd.read_csv(args.manifest)
        for _, row in df.iterrows():
            pid = row["patient_id"] if "patient_id" in row else Path(row["image_path"]).stem
            items.append((pid, row["image_path"]))
        return items

    if args.input_dir is not None:
        for p in sorted(Path(args.input_dir).glob("*")):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                items.append((p.stem, str(p)))
        return items

    raise ValueError("image veya manifest veya input-dir verilmelidir.")


def preprocess(img_path: str, image_size: int, mean, std):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {img_path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
    norm = (rgb - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = np.transpose(norm, (2, 0, 1))
    return img, torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)


def save_overlay(gray_uint8: np.ndarray, cam: np.ndarray, output_path: Path):
    hm = (cam * 255).astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    gray_bgr = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray_bgr, 0.6, hm, 0.4, 0)
    cv2.imwrite(str(output_path), overlay)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    map_dir = output_dir / "maps"
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    model, image_size, mean, std = load_model(args.checkpoint, args.device)
    gradcam = GradCAM(model)
    items = gather_items(args)

    rows = []
    for patient_id, img_path in tqdm(items, desc="Generating Grad-CAM"):
        raw, x = preprocess(img_path, image_size, mean, std)
        x = x.to(args.device)
        cam, prob = gradcam(x)

        cv2.imwrite(str(map_dir / f"{patient_id}.png"), (cam * 255).astype(np.uint8))
        save_overlay(raw, cam, overlay_dir / f"{patient_id}_overlay.png")

        rows.append(
            {
                "patient_id": patient_id,
                "img_path": img_path,
                "probability": prob,
                "cam_score": float(cam.mean()),
                "cam_max": float(cam.max()),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "gradcam_summary.csv", index=False)
    gradcam.close()

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"num_images": len(rows), "output_dir": str(output_dir)}, f, indent=2, ensure_ascii=False)

    print("[OK] Grad-CAM üretimi tamamlandı.")


if __name__ == "__main__":
    main()