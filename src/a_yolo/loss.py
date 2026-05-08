"""
A-YOLO Loss Function — Run 5 (100 Epoch) Optimized Version

Key Features:
  • GIoU (Generalized IoU) integration for precise bounding box regression.
  • Weighted BCE (pos_weight) to boost Recall/Sensitivity for pneumonia.
  • MAE-style masked reconstruction for SSL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class AYOLOLoss(nn.Module):
    """
    Args:
        alpha       : weight for SSL reconstruction loss (0.4 - 0.5 recommended for long runs)
        patch_size  : ViT patch size (default 16)
        pos_weight  : Boosts positive samples to increase Recall (default 2.5)
        lambda_reg  : weight for bounding box regression (default 3.0)
    """

    def __init__(self, alpha: float = 0.2, patch_size: int = 16,
                 pos_weight: float = 1.8, lambda_reg: float = 3.0):
        super().__init__()
        self.alpha      = alpha
        self.patch_size = patch_size
        self.lambda_reg = lambda_reg
        self.pos_weight_val = pos_weight
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.1)

    @staticmethod
    def patchify(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        """[B, C, H, W] -> [B, L, P^2 * C]"""
        p = patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h * w, p ** 2 * C)
        return x

    def forward(self,
                predictions: dict,
                targets: dict,
                original_imgs: torch.Tensor,
                mask_indices: torch.Tensor | None = None) -> dict:
        
        device = original_imgs.device
        B = original_imgs.shape[0]

        # ── 1. SSL Reconstruction Loss (MAE-style) ───────────────────────────
        target_patches = self.patchify(original_imgs, self.patch_size)
        recon          = predictions["reconstruction"]

        if mask_indices is not None:
            L = target_patches.shape[1]
            mask = torch.zeros(B, L, device=device)
            mask.scatter_(1, mask_indices, torch.ones(B, mask_indices.shape[1], device=device))
            sq_err     = (recon - target_patches) ** 2
            loss_recon = (sq_err.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss_recon = F.mse_loss(recon, target_patches)

        # ── 2. Classification Loss (Weighted for Recall) ─────────────────────
        pred_cls  = predictions["pred_cls"].view(-1)
        target_cls = targets["label"].view(-1).float().to(device)

        # 🚀 Pozitif (zatürre) vakaların cezasını artırarak Recall'u fırlatıyoruz
        pw        = torch.tensor([self.pos_weight_val], device=device)
        loss_cls  = F.binary_cross_entropy_with_logits(pred_cls, target_cls, pos_weight=pw)

        # ── 3. Regression Loss (SmoothL1 + GIoU Hybrid) ───────────────────────
        pos_mask = (targets["label"] > 0).view(-1)
        if pos_mask.any():
            pred_boxes = predictions["pred_reg"].view(-1, 4)[pos_mask]
            true_boxes = targets["bbox"].view(-1, 4)[pos_mask].to(device)
            
            # Distance loss (SmoothL1)
            l1_loss = self.smooth_l1(pred_boxes, true_boxes)
            
            # 🚀 GIoU Loss (Overlap loss)
            # torchvision GIoU [x1, y1, x2, y2] bekler. Koordinat dönüşümü:
            def xywh_to_xyxy(boxes):
                x, y, w, h = boxes.unbind(-1)
                return torch.stack([x, y, x + w, y + h], dim=-1)
            
            # generalized_box_iou bir matris döner, biz sadece çapraz elemanları (eşleşenleri) alıyoruz
            giou_matrix = ops.generalized_box_iou(xywh_to_xyxy(pred_boxes), xywh_to_xyxy(true_boxes))
            giou_loss   = 1.0 - torch.diag(giou_matrix).mean()
            
            # Hibrit regreasyon: Hem köşeler yaklaşsın hem de kutular örtüşsün
            loss_reg = 0.5 * l1_loss + 0.5 * giou_loss
        else:
            # Negatif batch durumunda gradyanı bozmamak için sıfır döndür
            loss_reg = predictions["pred_reg"].sum() * 0.0

        # ── 4. Combined Loss ─────────────────────────────────────────────────
        loss_det   = loss_cls + self.lambda_reg * loss_reg
        total_loss = self.alpha * loss_recon + (1 - self.alpha) * loss_det

        return {
            "total_loss": total_loss,
            "recon_loss": loss_recon.detach(),
            "cls_loss":   loss_cls.detach(),
            "reg_loss":   loss_reg.detach(),
        }