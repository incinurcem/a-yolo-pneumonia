"""
A-YOLO: Anatomy-Aware Masked YOLO
Model Architecture: ViT-Base backbone + MAE decoder + Detection Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import vit_base_patch16_224


class AYOLO(nn.Module):
    """
    Anatomy-Aware Masked YOLO:
    - Backbone  : ViT-Base/16 (pretrained ImageNet, 3-channel RGB)
    - SSL Head  : MAE-style masked patch reconstruction
    - Det Head  : Decoupled cls + reg (GAP -> MLP)

    Forward args:
        x             : [B, 3, H, W] normalized image tensor
        mask_indices  : [B, num_masked]  long tensor  (None at eval time)

    Returns dict:
        reconstruction : [B, L, patch_size^2 * 3]
        pred_cls       : [B, 1]  (raw logit, apply sigmoid outside)
        pred_reg       : [B, 4]  (sigmoid-normalized [x,y,w,h] in [0,1])
        features       : [B, L, C]  patch-level features for explainability
    """

    def __init__(self, num_classes: int = 1, patch_size: int = 16,
                 img_size: int = 224, drop_rate: float = 0.1):
        super().__init__()

        # ── 1. ViT Backbone (3-channel, ImageNet pretrained) ──────────────────
        self.backbone = vit_base_patch16_224(pretrained=True)
        self.embed_dim = self.backbone.embed_dim   # 768
        self.patch_size = patch_size
        self.grid_size  = img_size // patch_size   # 14
        self.num_patches = self.grid_size ** 2     # 196

        # Remove the backbone's own classification head – we use ours
        self.backbone.head = nn.Identity()

        # ── 2. SSL / MAE Components ───────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Pixel-space decoder  →  [B, L, P² · 3]
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, patch_size ** 2 * 3),
        )

        # ── 3. Detection Head (Decoupled) ─────────────────────────────────────
        self.det_norm = nn.LayerNorm(self.embed_dim)

        # Classification: pneumonia present?
        self.cls_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
            # NO sigmoid here → applied in loss (BCEWithLogits) and inference
        )

        # Regression: [x, y, w, h] all in [0, 1]
        self.reg_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 4),
            nn.Sigmoid(),   # ← Critical: guarantees valid bbox coords
        )

        # ── Weight Init ───────────────────────────────────────────────────────
        nn.init.normal_(self.mask_token, std=0.02)
        self._init_detection_weights()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_detection_weights(self):
        """Xavier-uniform init for detection head linears only."""
        for m in list(self.cls_head.modules()) + list(self.reg_head.modules()) + \
                 list(self.decoder.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor,
                mask_indices: torch.Tensor | None = None) -> dict:
        """
        x            : [B, 3, H, W]
        mask_indices : [B, num_masked]  LongTensor  (None → no masking)
        """
        B = x.shape[0]

        # A) Patch embedding (no CLS token yet, position embed on patches)
        x_patch = self.backbone.patch_embed(x)          # [B, 196, 768]
        x_patch = x_patch + self.backbone.pos_embed[:, 1:, :]  # skip cls pos

        # B) Anatomy-Aware Masking ────────────────────────────────────────────
        if mask_indices is not None:
            mask_tokens = self.mask_token.expand(B, x_patch.shape[1], -1)
            # Vectorised mask construction (no Python loop)
            w = torch.zeros(B, x_patch.shape[1], 1, device=x.device)
            # scatter_  sets w[b, mask_indices[b], 0] = 1  for each b
            w.scatter_(1,
                       mask_indices.unsqueeze(-1),   # [B, num_masked, 1]
                       torch.ones(B, mask_indices.shape[1], 1,
                                  device=x.device))
            x_patch = x_patch * (1 - w) + mask_tokens * w

        # C) Transformer Encoder ──────────────────────────────────────────────
        for blk in self.backbone.blocks:
            x_patch = blk(x_patch)
        x_patch = self.backbone.norm(x_patch)           # [B, 196, 768]

        # D) SSL / Reconstruction output ──────────────────────────────────────
        recon_out = self.decoder(x_patch)               # [B, 196, P²·3]

        # E) Detection output (Global Average Pooling) ────────────────────────
        feat_gap = x_patch.mean(dim=1)                  # [B, 768]
        feat_gap = self.det_norm(feat_gap)

        pred_cls = self.cls_head(feat_gap)              # [B, 1]
        pred_reg = self.reg_head(feat_gap)              # [B, 4]  ∈ [0,1]

        return {
            "reconstruction": recon_out,
            "pred_cls":        pred_cls,
            "pred_reg":        pred_reg,
            "features":        x_patch,     # for Grad-CAM hooks
        }
