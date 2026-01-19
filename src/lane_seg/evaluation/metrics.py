import torch


@torch.no_grad()
def dice_from_logits(logits, targets, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    preds_f = preds.view(preds.size(0), -1)
    t_f = targets.view(targets.size(0), -1)
    inter = (preds_f * t_f).sum(dim=1)
    return (2 * inter + eps) / (preds_f.sum(dim=1) + t_f.sum(dim=1) + eps)


@torch.no_grad()
def iou_from_logits(logits, targets, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    preds_f = preds.view(preds.size(0), -1)
    t_f = targets.view(targets.size(0), -1)
    inter = (preds_f * t_f).sum(dim=1)
    union = preds_f.sum(dim=1) + t_f.sum(dim=1) - inter
    return (inter + eps) / (union + eps)


@torch.no_grad()
def prf_from_logits(logits, targets, thr=0.5, eps=1e-7):
    """
    Pixel-wise Precision / Recall / F1 for foreground(1) class.
    logits:  [B,1,H,W]
    targets: [B,1,H,W] (0/1)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    preds_f = preds.view(preds.size(0), -1)
    t_f = targets.view(targets.size(0), -1)

    tp = (preds_f * t_f).sum(dim=1)
    fp = (preds_f * (1 - t_f)).sum(dim=1)
    fn = ((1 - preds_f) * t_f).sum(dim=1)

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return precision, recall, f1
