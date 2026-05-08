import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

from box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcher(nn.Module):
    """
    RSNA Pneumonia Detection için optimize edilmiş Hungarian Matcher.
    
    Önemli Güncellemeler:
    1. Sigmoid-based Focal Cost: SetCriterion'daki Focal Loss ile tam uyumlu eşleşme sağlar.
    2. Cost Balancing: Sınıflandırma maliyeti (cost_class) artırılarak modelin 
       "hiç kutu çizmeme" eğilimi kırılmıştır.
    """
    def __init__(
        self,
        cost_class: float = 5.0,  # Lezyon bulmaya zorlamak için yüksek tutuldu
        cost_bbox: float = 2.0,   # Kutu hassasiyeti başlangıçta esnetildi
        cost_giou: float = 1.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("Tüm maliyet (cost) katsayıları 0 olamaz.")

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Hungarian matching algoritmasını Focal Loss maliyetleriyle çalıştırır.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 1. Tahminleri Hazırla - [B*Q, C] ve [B*Q, 4]
        # Focal Loss sigmoid tabanlı olduğu için burada softmax yerine sigmoid kullanıyoruz.
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # 2. Hedefleri Hazırla
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 3. Sınıflandırma Maliyeti (Focal Cost)
        # Focal Loss'un matematiksel maliyetini Matcher'a öğretiyoruz.
        # Bu, modelin "kolay" arka plan örneklerini değil, "zor" lezyonları eşleştirmesini sağlar.
        alpha = 0.25
        gamma = 2.0
        
        # Sadece lezyon kanalı (indeks 0) için pozitif ve negatif maliyetler
        # focal_cost = -(alpha * (1-p)^gamma * log(p) - (1-alpha) * p^gamma * log(1-p))
        # Basitleştirilmiş hali:
        pos_cost_class = -(1 - out_prob[:, tgt_ids]) ** gamma * out_prob[:, tgt_ids].add(1e-8).log() * alpha
        neg_cost_class = -out_prob[:, tgt_ids] ** gamma * (1 - out_prob[:, tgt_ids]).add(1e-8).log() * (1 - alpha)
        
        # Son maliyet: Pozitif eşleşmenin avantajı - Negatifin avantajı
        cost_class = pos_cost_class - neg_cost_class

        # 4. Box L1 Maliyeti
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 5. GIoU Maliyeti
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )

        # 6. Toplam Maliyet Matrisi (C)
        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )
        
        # Batch yapısına geri dön: [B, num_queries, toplam_hedef_sayisi]
        C = C.view(bs, num_queries, -1).cpu()

        # Batch içindeki her bir resmin kaç hedefi olduğunu bul
        sizes = [len(v["boxes"]) for v in targets]
        
        indices = []
        start = 0
        for b, size in enumerate(sizes):
            if size == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64)
                ))
                continue

            # Mevcut resme ait maliyet dilimi
            c = C[b, :, start : start + size]
            # Macar Algoritması (Hungarian) - Bellek dostu olması için numpy/scipy kullanır
            src_idx, tgt_idx = linear_sum_assignment(c)
            
            indices.append((
                torch.as_tensor(src_idx, dtype=torch.int64),
                torch.as_tensor(tgt_idx, dtype=torch.int64)
            ))
            start += size

        return indices