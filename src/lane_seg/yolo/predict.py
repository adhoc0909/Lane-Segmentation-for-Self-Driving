from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class YoloMaskPrediction:
    """Binary union mask derived from YOLO instance masks."""

    binary_mask: np.ndarray  # uint8 0/1, shape (H, W)
    conf_max: float


def load_yolo_model(weights: str):
    """Load Ultralytics YOLO model (lazy dependency)."""
    from ultralytics import YOLO

    return YOLO(weights)


def yolo_predict_union_mask(
    model,
    img_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.7,
    classes: Optional[List[int]] = None,
    imgsz: Optional[int] = None,
) -> YoloMaskPrediction:
    """Predict and convert instance masks to a single binary mask.

    - For lane semantic mask usage, we take the union of all instance masks.
    - Returns mask in original image resolution.
    """
    h, w = img_bgr.shape[:2]

    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        classes=classes,
        imgsz=imgsz,
        verbose=False,
    )

    if not results:
        return YoloMaskPrediction(binary_mask=np.zeros((h, w), dtype=np.uint8), conf_max=0.0)

    r = results[0]
    if getattr(r, "masks", None) is None:
        return YoloMaskPrediction(binary_mask=np.zeros((h, w), dtype=np.uint8), conf_max=0.0)

    # r.masks.data: [N, Hm, Wm] float tensor on GPU/CPU, already scaled to original image
    m = r.masks.data
    try:
        m = m.detach().float().cpu().numpy()
    except Exception:
        m = np.asarray(m)

    if m.ndim != 3 or m.shape[0] == 0:
        return YoloMaskPrediction(binary_mask=np.zeros((h, w), dtype=np.uint8), conf_max=0.0)

    # Ensure mask resolution matches image; if not, resize each
    union = np.zeros((h, w), dtype=np.uint8)
    for i in range(m.shape[0]):
        mi = m[i]
        if mi.shape[0] != h or mi.shape[1] != w:
            mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_NEAREST)
        union = np.maximum(union, (mi > 0.5).astype(np.uint8))

    conf_max = 0.0
    try:
        if getattr(r, "boxes", None) is not None and r.boxes.conf is not None:
            conf_max = float(r.boxes.conf.max().item()) if len(r.boxes.conf) else 0.0
    except Exception:
        pass

    return YoloMaskPrediction(binary_mask=union, conf_max=conf_max)


def overlay_mask_on_bgr(
    img_bgr: np.ndarray,
    mask01: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary mask on a BGR image."""
    out = img_bgr.copy()
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)
    mask = (mask01 > 0)
    if not mask.any():
        return out

    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[mask] = color
    out = cv2.addWeighted(out, 1.0, overlay, alpha, 0)
    return out


def save_mask_png(mask01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m = (mask01.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), m)
