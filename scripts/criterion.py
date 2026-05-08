from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class SetCriterion(nn.Module):
    """
    RSNA Pneumonia Detection için optimize edilmiş Kayıp Fonksiyonu.
    
    Kritik İyileştirmeler:
    1. Sigmoid Focal Loss: Sınıf dengesizliğini cerrahi bir hassasiyetle çözer.
    2. Kademeli Aux Loss: Yardımcı katmanların gürültüsünü azaltarak Val Loss'u aşağı çeker.
    3. Stabil Normalizasyon: num_boxes hesaplamasını güvenli hale getirir.
    """
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        eos_coef: float,
        losses: List[str]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # Boş sınıf (background) ağırlığı (Sadece Cross Entropy kalıntıları için)
        empty_weight = torch.ones(num_classes)
        if num_classes > 0:
            empty_weight[num_classes - 1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def sigmoid_focal_loss(self, inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0):
        """
        Matematiksel Formül: 
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.sum() / max(num_boxes, 1.0)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices,
        num_boxes: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> Dict[str, torch.Tensor]:
        """
        Sınıflandırma Kaybı: Focal Loss ile 'körlük' engellenir.
        0: Lesion, 1: Background
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"] # [B, Q, C]

        idx = self._get_src_permutation_idx(indices)
        
        # Target oluşturma (Background: 1, Lesion: 0)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes - 1,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = 0 

        # Focal Loss için One-Hot (Sadece ön plan sınıfları için)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes],
                                           dtype=src_logits.dtype, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        
        # Sadece lezyon kanalını (index 0) eğitiyoruz, background 'sessizlik' olarak kalıyor
        target_classes_onehot = target_classes_onehot[:, :, :self.num_classes-1]
        src_logits_fg = src_logits[:, :, :self.num_classes-1]

        loss_ce = self.sigmoid_focal_loss(src_logits_fg, target_classes_onehot, num_boxes, alpha, gamma)
        
        losses = {"loss_ce": loss_ce}

        # Monitoring için hata payı
        with torch.no_grad():
            losses["class_error"] = 100 - (src_logits.argmax(-1) == target_classes).float().mean() * 100
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Regresyon Kaybı: L1 + GIoU
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {"loss_bbox": torch.tensor(0.0, device=num_boxes.device), 
                    "loss_giou": torch.tensor(0.0, device=num_boxes.device)}

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), 
                                                      box_cxcywh_to_xyxy(target_boxes)))

        return {
            "loss_bbox": loss_bbox.sum() / max(num_boxes, 1.0),
            "loss_giou": loss_giou.sum() / max(num_boxes, 1.0)
        }

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Tahmin edilen kutu sayısı takibi """
        pred_logits = outputs["pred_logits"]
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=pred_logits.device)
        card_pred = (pred_logits.argmax(-1) != self.num_classes - 1).sum(1)
        return {"cardinality_error": F.l1_loss(card_pred.float(), tgt_lengths.float())}

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {"labels": self.loss_labels, "boxes": self.loss_boxes, "cardinality": self.loss_cardinality}
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        Forward Pass: Loss hesaplamasının kalbi.
        """
        # Ana katman eşleşmesi
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        # num_boxes normalizasyonu
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["pred_logits"].device)
        num_boxes = torch.clamp(num_boxes, min=1.0)

        # Ana katman kayıpları
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # YARDIMCI KATMANLAR (Deep Supervision)
        # Loss 10 barajını yıkan stratejik ağırlıklandırma
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_outputs, targets)
                # Gürültüyü azaltmak için aux katmanların ağırlığını kıstık (0.5)
                # İlk katmanlar daha az, son katman daha çok güvenilirdir.
                layer_weight = 0.5 
                
                for loss in self.losses:
                    if loss == "cardinality": continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes)
                    l_dict = {f"{k}_{i}": v * layer_weight for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses