import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


# ============================================================
# 1. YARDIMCI BLOKLAR
# ============================================================

def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.GroupNorm(num_groups=8 if out_ch >= 8 else 1, num_channels=out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        return x


# ============================================================
# 2. POSITION EMBEDDING
# ============================================================

class PositionEmbeddingSine2D(nn.Module):
    """
    DETR benzeri 2D sine-cosine positional embedding.
    Girdi: [B, C, H, W]
    Çıktı: [B, hidden_dim, H, W]
    """
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True, scale: float = 2 * math.pi):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, _, h, w = x.shape
        if mask is None:
            mask = torch.zeros((b, h, w), dtype=torch.bool, device=x.device)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        return pos


# ============================================================
# 3. SWIN BACKBONE WRAPPER
# ============================================================

class SwinBackboneWrapper(nn.Module):
    """
    timm tabanlı Swin feature extractor.
    Grayscale input'u 1->3 kanal çevirerek kullanır.
    Multi-scale feature döndürür.
    """
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window12_384",
        pretrained: bool = True,
        in_chans: int = 1,
        image_size: int = 384,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3)
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm kurulu değil. Lütfen `pip install timm` yap.")

        self.in_chans = in_chans
        self.image_size = image_size
        self.model_name = model_name

        if in_chans not in [1, 3]:
            raise ValueError(f"SwinBackboneWrapper yalnızca 1 veya 3 giriş kanalını destekliyor. Gelen: {in_chans}")

        # Pretrained backbone ile en güvenli kullanım:
        # grayscale görüntüyü 1->3 çevirip timm backbone'u 3 kanallı kurmak.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            img_size=image_size,
            in_chans=3,
        )
        self.feature_channels = self.backbone.feature_info.channels()

        if in_chans == 1:
            self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.input_adapter = nn.Identity()

    def _ensure_nchw(self, feat: torch.Tensor, expected_channels: int) -> torch.Tensor:
        """
        Bazı timm feature extractor sürümlerinde Swin çıktıları NHWC gelebilir.
        Bunları güvenli şekilde NCHW'ye çeviriyoruz.
        """
        if feat.ndim != 4:
            raise ValueError(f"Backbone feature 4D olmalı, gelen shape: {tuple(feat.shape)}")

        # Zaten NCHW ise
        if feat.shape[1] == expected_channels:
            return feat.contiguous()

        # NHWC ise
        if feat.shape[-1] == expected_channels:
            return feat.permute(0, 3, 1, 2).contiguous()

        raise ValueError(
            f"Backbone feature channel uyuşmazlığı. "
            f"Beklenen kanal: {expected_channels}, gelen shape: {tuple(feat.shape)}"
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: [B, 1, H, W] veya [B, 3, H, W]
        returns: list of features
            f0: [B, C0, H/4,  W/4]
            f1: [B, C1, H/8,  W/8]
            f2: [B, C2, H/16, W/16]
            f3: [B, C3, H/32, W/32]
        """
        if x.ndim != 4:
            raise ValueError(f"Backbone input 4D olmalı. Gelen shape: {tuple(x.shape)}")

        if x.shape[1] not in [1, 3]:
            raise ValueError(f"Backbone input kanal sayısı 1 veya 3 olmalı. Gelen: {x.shape[1]}")

        x = self.input_adapter(x)
        feats = self.backbone(x)

        feats = [
            self._ensure_nchw(feat, expected_channels=ch)
            for feat, ch in zip(feats, self.feature_channels)
        ]
        return feats


# ============================================================
# 4. DIFFUSION / STRUCTURE ENCODER
# ============================================================

class StructureResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvGNAct(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8 if ch >= 8 else 1, num_channels=ch)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


class DiffusionStructureEncoder(nn.Module):
    """
    Tam diffusion eğitimi yapmayan ama diffusion encoder fikrini taklit eden
    hafif UNet-benzeri structure prior encoder.
    """
    def __init__(
        self,
        in_chans: int = 1,
        base_dim: int = 32,
        out_dims: Tuple[int, ...] = (96, 192, 384, 768)
    ):
        super().__init__()

        self.stem = nn.Sequential(
            ConvGNAct(in_chans, base_dim, k=3, s=1, p=1),
            StructureResidualBlock(base_dim)
        )

        self.stage1 = nn.Sequential(
            ConvGNAct(base_dim, base_dim * 2, k=3, s=2, p=1),
            StructureResidualBlock(base_dim * 2),
            StructureResidualBlock(base_dim * 2)
        )
        self.stage2 = nn.Sequential(
            ConvGNAct(base_dim * 2, base_dim * 4, k=3, s=2, p=1),
            StructureResidualBlock(base_dim * 4),
            StructureResidualBlock(base_dim * 4)
        )
        self.stage3 = nn.Sequential(
            ConvGNAct(base_dim * 4, base_dim * 8, k=3, s=2, p=1),
            StructureResidualBlock(base_dim * 8),
            StructureResidualBlock(base_dim * 8)
        )
        self.stage4 = nn.Sequential(
            ConvGNAct(base_dim * 8, base_dim * 16, k=3, s=2, p=1),
            StructureResidualBlock(base_dim * 16),
            StructureResidualBlock(base_dim * 16)
        )

        self.out_projs = nn.ModuleList([
            nn.Conv2d(base_dim * 2, out_dims[0], kernel_size=1),
            nn.Conv2d(base_dim * 4, out_dims[1], kernel_size=1),
            nn.Conv2d(base_dim * 8, out_dims[2], kernel_size=1),
            nn.Conv2d(base_dim * 16, out_dims[3], kernel_size=1),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        s1 = self.stage1(x)   # /2
        s2 = self.stage2(s1)  # /4
        s3 = self.stage3(s2)  # /8
        s4 = self.stage4(s3)  # /16

        d1 = self.out_projs[0](s1)
        d2 = self.out_projs[1](s2)
        d3 = self.out_projs[2](s3)
        d4 = self.out_projs[3](s4)

        return [d1, d2, d3, d4]


# ============================================================
# 5. GUIDANCE FUSION BLOCK
# ============================================================

class GuidanceFusionBlock(nn.Module):
    """
    Backbone feature F ile diffusion prior D'yi birleştirir.

    Desteklenen modlar:
    - add
    - gate
    - hybrid
    """
    def __init__(
        self,
        feat_dim: int,
        diff_dim: int,
        mode: str = "hybrid",
        use_layernorm: bool = True
    ):
        super().__init__()
        assert mode in ["add", "gate", "hybrid"]

        self.mode = mode
        self.diff_proj = nn.Sequential(
            nn.Conv2d(diff_dim, feat_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8 if feat_dim >= 8 else 1, num_channels=feat_dim),
            nn.GELU()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.post = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8 if feat_dim >= 8 else 1, num_channels=feat_dim),
            nn.GELU()
        )

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(feat_dim)

    def forward(self, feat: torch.Tensor, diff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if diff.shape[-2:] != feat.shape[-2:]:
            diff = F.interpolate(diff, size=feat.shape[-2:], mode="bilinear", align_corners=False)

        d_proj = self.diff_proj(diff)
        gate = torch.sigmoid(d_proj)

        if self.mode == "add":
            fused = feat + self.alpha * d_proj
        elif self.mode == "gate":
            fused = feat * gate
        else:
            fused = feat + self.alpha * d_proj + self.beta * (feat * gate)

        fused = self.post(fused)

        if self.use_layernorm:
            b, c, h, w = fused.shape
            fused = fused.permute(0, 2, 3, 1).contiguous()
            fused = self.ln(fused)
            fused = fused.permute(0, 3, 1, 2).contiguous()

        guidance_map = gate.mean(dim=1, keepdim=True)
        return fused, guidance_map


class MultiScaleGuidanceFusion(nn.Module):
    def __init__(
        self,
        backbone_dims: List[int],
        diffusion_dims: List[int],
        mode: str = "hybrid"
    ):
        super().__init__()
        assert len(backbone_dims) == len(diffusion_dims)

        self.blocks = nn.ModuleList([
            GuidanceFusionBlock(feat_dim=fd, diff_dim=dd, mode=mode)
            for fd, dd in zip(backbone_dims, diffusion_dims)
        ])

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        diffusion_feats: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        fused_feats = []
        guidance_maps = []

        if len(diffusion_feats) < len(backbone_feats):
            repeat_count = len(backbone_feats) - len(diffusion_feats)
            diffusion_feats = [diffusion_feats[0]] * repeat_count + diffusion_feats
        elif len(diffusion_feats) > len(backbone_feats):
            diffusion_feats = diffusion_feats[-len(backbone_feats):]

        for feat, diff, block in zip(backbone_feats, diffusion_feats, self.blocks):
            fused, gmap = block(feat, diff)
            fused_feats.append(fused)
            guidance_maps.append(gmap)

        return fused_feats, guidance_maps


# ============================================================
# 6. MULTI-SCALE FEATURE PROJECTION
# ============================================================

class MultiScaleFeatureProjector(nn.Module):
    def __init__(self, in_dims: List[int], hidden_dim: int, num_feature_levels: int = 4):
        super().__init__()
        self.num_feature_levels = num_feature_levels

        self.input_proj = nn.ModuleList()
        for in_dim in in_dims:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32 if hidden_dim >= 32 else 1, hidden_dim)
                )
            )

        self.extra_proj = nn.ModuleList()
        extra_levels = max(0, num_feature_levels - len(in_dims))
        for _ in range(extra_levels):
            self.extra_proj.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32 if hidden_dim >= 32 else 1, hidden_dim)
                )
            )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for feat, proj in zip(feats, self.input_proj):
            out.append(proj(feat))

        x = out[-1]
        for proj in self.extra_proj:
            x = proj(x)
            out.append(x)

        return out


# ============================================================
# 7. DEFORMABLE CROSS-ATTENTION
# ============================================================

class MultiScaleDeformableCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.d_per_head = hidden_dim // n_heads

        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.sampling_offsets = nn.Linear(hidden_dim, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(hidden_dim, n_heads * n_levels * n_points)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)

        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (i + 1)

        self.sampling_offsets.bias = nn.Parameter(grid_init.flatten())
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        multi_level_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        b, nq, c = query.shape
        assert len(multi_level_feats) == self.n_levels

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            b, nq, self.n_heads, self.n_levels, self.n_points, 2
        )

        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            b, nq, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            b, nq, self.n_heads, self.n_levels, self.n_points
        )

        ref = reference_points[:, :, None, None, None, :]

        sampled_all_levels = []

        for lvl, feat in enumerate(multi_level_feats):
            b_, c_, h, w = feat.shape
            assert b_ == b and c_ == c

            value = feat.flatten(2).transpose(1, 2)
            value = self.value_proj(value)
            value = value.transpose(1, 2).view(b, self.n_heads, self.d_per_head, h, w)

            offsets_lvl = sampling_offsets[:, :, :, lvl, :, :]

            offset_norm = torch.zeros_like(offsets_lvl)
            offset_norm[..., 0] = offsets_lvl[..., 0] / max(w, 1)
            offset_norm[..., 1] = offsets_lvl[..., 1] / max(h, 1)

            sampling_locations = ref[:, :, :, :, 0, :] + offset_norm
            sampling_grid = sampling_locations * 2.0 - 1.0

            sampled_heads = []

            for head_idx in range(self.n_heads):
                value_h = value[:, head_idx]
                grid_h = sampling_grid[:, :, head_idx, :, :]

                sampled = F.grid_sample(
                    value_h,
                    grid_h,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False
                )

                sampled = sampled.permute(0, 2, 3, 1).contiguous()
                sampled_heads.append(sampled)

            sampled_heads = torch.stack(sampled_heads, dim=2)
            sampled_all_levels.append(sampled_heads)

        sampled_all_levels = torch.stack(sampled_all_levels, dim=3)
        attn = attention_weights.unsqueeze(-1)
        out = (sampled_all_levels * attn).sum(dim=4).sum(dim=3)
        out = out.reshape(b, nq, c)
        out = self.output_proj(out)
        out = self.dropout(out)
        return out


# ============================================================
# 8. DECODER LAYER
# ============================================================

class DeformableDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MultiScaleDeformableCrossAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_levels=n_levels,
            n_points=n_points,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim)
        )

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        multi_level_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            query=tgt + query_pos,
            reference_points=reference_points,
            multi_level_feats=multi_level_feats
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# ============================================================
# 9. DEFORMABLE DETR DECODER
# ============================================================

class DeformableDETRDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.num_classes = num_classes

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        self.reference_points = nn.Linear(hidden_dim, 2)

        self.layers = nn.ModuleList([
            DeformableDecoderLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_levels=n_levels,
                n_points=n_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, multi_level_feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        b = multi_level_feats[0].shape[0]

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        tgt = query_embed

        init_reference = torch.sigmoid(self.reference_points(query_pos))
        reference_points = init_reference

        inter_states = []
        inter_references = []

        for layer in self.layers:
            tgt = layer(
                tgt=tgt,
                query_pos=query_pos,
                reference_points=reference_points,
                multi_level_feats=multi_level_feats
            )

            delta_ref = self.reference_points(tgt)
            reference_points = torch.sigmoid(delta_ref + inverse_sigmoid(reference_points))

            inter_states.append(tgt)
            inter_references.append(reference_points)

        inter_states = torch.stack(inter_states, dim=0)
        inter_references = torch.stack(inter_references, dim=0)

        outputs_classes = []
        outputs_coords = []

        for lvl in range(inter_states.shape[0]):
            hs = inter_states[lvl]
            ref = inter_references[lvl]

            cls = self.class_embed(hs)
            box_delta = self.bbox_embed(hs)

            box = box_delta.clone()
            box[..., :2] = torch.sigmoid(box_delta[..., :2] + inverse_sigmoid(ref))
            box[..., 2:] = torch.sigmoid(box_delta[..., 2:])

            outputs_classes.append(cls)
            outputs_coords.append(box)

        outputs_class = torch.stack(outputs_classes, dim=0)
        outputs_coord = torch.stack(outputs_coords, dim=0)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "aux_outputs": [
                {"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ],
            "decoder_hidden_states": inter_states,
            "reference_points": inter_references,
        }
        return out


# ============================================================
# 10. LIGHT ENCODER FOR FUSED FEATURES
# ============================================================

class MultiScaleTransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, feats: List[torch.Tensor], pos_embeds: List[torch.Tensor]) -> List[torch.Tensor]:
        out_feats = []
        for feat, pos in zip(feats, pos_embeds):
            b, c, h, w = feat.shape
            x = feat.flatten(2).transpose(1, 2)
            p = pos.flatten(2).transpose(1, 2)
            x = self.encoder(x + p)
            x = x.transpose(1, 2).reshape(b, c, h, w)
            out_feats.append(x)
        return out_feats


# ============================================================
# 11. MODEL OUTPUT DATACLASS
# ============================================================

@dataclass
class DiffusionGuidedDetrOutput:
    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor
    aux_outputs: List[Dict[str, torch.Tensor]]
    backbone_features: List[torch.Tensor]
    diffusion_priors: List[torch.Tensor]
    fused_features: List[torch.Tensor]
    projected_features: List[torch.Tensor]
    guidance_maps: List[torch.Tensor]
    decoder_hidden_states: torch.Tensor
    reference_points: torch.Tensor
    losses: Optional[Dict[str, torch.Tensor]] = None


# ============================================================
# 12. ANA MODEL
# ============================================================

class DiffusionGuidedDeformableDETR(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 384,
        num_queries: int = 300,
        hidden_dim: int = 256,
        num_feature_levels: int = 4,
        backbone_name: str = "swin_tiny_patch4_window12_384",
        backbone_pretrained: bool = True,
        fusion_mode: str = "hybrid",
        decoder_layers: int = 6,
        encoder_layers: int = 2,
        n_heads: int = 8,
        n_points: int = 4,
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.criterion = criterion

        self.backbone = SwinBackboneWrapper(
            model_name=backbone_name,
            pretrained=backbone_pretrained,
            in_chans=1,
            image_size=image_size,
            out_indices=(0, 1, 2, 3)
        )
        backbone_dims = list(self.backbone.feature_channels)

        self.diffusion_encoder = DiffusionStructureEncoder(
            in_chans=1,
            base_dim=32,
            out_dims=tuple(backbone_dims)
        )

        self.guidance_fusion = MultiScaleGuidanceFusion(
            backbone_dims=backbone_dims,
            diffusion_dims=backbone_dims,
            mode=fusion_mode
        )

        self.projector = MultiScaleFeatureProjector(
            in_dims=backbone_dims,
            hidden_dim=hidden_dim,
            num_feature_levels=num_feature_levels
        )

        self.position_embedding = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2)

        self.encoder = MultiScaleTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            n_heads=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, hidden_dim))
        nn.init.normal_(self.level_embed)

        self.decoder = DeformableDETRDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=decoder_layers,
            n_heads=n_heads,
            n_levels=num_feature_levels,
            n_points=n_points,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            num_classes=num_classes,
        )

    def _build_positional_embeddings(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        pos_embeds = []
        for lvl, feat in enumerate(feats):
            pos = self.position_embedding(feat)
            pos = pos + self.level_embed[lvl].view(1, -1, 1, 1)
            pos_embeds.append(pos)
        return pos_embeds

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> DiffusionGuidedDetrOutput:
        backbone_feats = self.backbone(images)

        diffusion_priors = self.diffusion_encoder(images)

        if len(diffusion_priors) != len(backbone_feats):
            if len(diffusion_priors) > len(backbone_feats):
                diffusion_priors = diffusion_priors[-len(backbone_feats):]
            else:
                diffusion_priors = [diffusion_priors[0]] * (len(backbone_feats) - len(diffusion_priors)) + diffusion_priors

        fused_feats, guidance_maps = self.guidance_fusion(backbone_feats, diffusion_priors)

        projected_feats = self.projector(fused_feats)

        pos_embeds = self._build_positional_embeddings(projected_feats)

        encoded_feats = self.encoder(projected_feats, pos_embeds)

        detr_out = self.decoder(encoded_feats)

        losses = None
        if self.criterion is not None and targets is not None:
            losses = self.criterion(
                outputs={
                    "pred_logits": detr_out["pred_logits"],
                    "pred_boxes": detr_out["pred_boxes"],
                    "aux_outputs": detr_out["aux_outputs"],
                },
                targets=targets
            )

        return DiffusionGuidedDetrOutput(
            pred_logits=detr_out["pred_logits"],
            pred_boxes=detr_out["pred_boxes"],
            aux_outputs=detr_out["aux_outputs"],
            backbone_features=backbone_feats,
            diffusion_priors=diffusion_priors,
            fused_features=fused_feats,
            projected_features=encoded_feats,
            guidance_maps=guidance_maps,
            decoder_hidden_states=detr_out["decoder_hidden_states"],
            reference_points=detr_out["reference_points"],
            losses=losses
        )


# ============================================================
# 13. BASİT KULLANIM TESTİ
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiffusionGuidedDeformableDETR(
        num_classes=2,
        image_size=384,
        num_queries=300,
        hidden_dim=256,
        num_feature_levels=4,
        backbone_name="swin_tiny_patch4_window12_384",
        backbone_pretrained=False,
        fusion_mode="hybrid",
        decoder_layers=6,
        encoder_layers=2,
        n_heads=8,
        n_points=4,
        criterion=None
    ).to(device)

    x = torch.randn(2, 1, 384, 384).to(device)
    out = model(x)

    print("pred_logits:", out.pred_logits.shape)
    print("pred_boxes :", out.pred_boxes.shape)
    print("num aux    :", len(out.aux_outputs))
    print("guidance   :", [g.shape for g in out.guidance_maps])