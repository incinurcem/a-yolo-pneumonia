import torch


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 4] format = (cx, cy, w, h), normalized ya da pixel olabilir
    returns: [..., 4] format = (x0, y0, x1, y1)
    """
    cx, cy, w, h = x.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 4] format = (x0, y0, x1, y1)
    returns: [..., 4] format = (cx, cy, w, h)
    """
    x0, y0, x1, y1 = x.unbind(-1)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = (x1 - x0)
    h = (y1 - y0)
    return torch.stack([cx, cy, w, h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: [N, 4] in xyxy
    returns: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    boxes1: [N, 4] xyxy
    boxes2: [M, 4] xyxy

    returns:
        iou: [N, M]
        union: [N, M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N, M, 2]

    wh = (rb - lt).clamp(min=0)                          # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                   # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: [N, 4] xyxy
    boxes2: [M, 4] xyxy

    returns:
        giou: [N, M]
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 xyxy format invalid"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 xyxy format invalid"

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area.clamp(min=1e-6)


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    masks: [N, H, W]
    returns: [N, 4] xyxy
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float32)

    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float32, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float32, device=masks.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * xx.unsqueeze(0)
    y_mask = masks * yy.unsqueeze(0)

    x_max = x_mask.flatten(1).max(-1)[0]
    y_max = y_mask.flatten(1).max(-1)[0]

    x_min = x_mask.masked_fill(~masks.bool(), float("inf")).flatten(1).min(-1)[0]
    y_min = y_mask.masked_fill(~masks.bool(), float("inf")).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], dim=1)