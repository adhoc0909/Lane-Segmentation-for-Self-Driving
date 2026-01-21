from __future__ import annotations

import numpy as np


def dice_iou_prf_from_masks(pred01: np.ndarray, tgt01: np.ndarray, eps: float = 1e-7):
    """Compute Dice, IoU, Precision, Recall, F1 from binary masks.

    pred01, tgt01: uint8/bool arrays of shape (H, W) or (1, H, W)
    Returns: (dice, iou, precision, recall, f1)
    """
    p = np.asarray(pred01)
    t = np.asarray(tgt01)

    if p.ndim == 3:
        p = p.squeeze(0)
    if t.ndim == 3:
        t = t.squeeze(0)

    p = (p > 0).astype(np.uint8)
    t = (t > 0).astype(np.uint8)

    tp = float(np.sum(p * t))
    fp = float(np.sum(p * (1 - t)))
    fn = float(np.sum((1 - p) * t))

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    return dice, iou, precision, recall, f1
