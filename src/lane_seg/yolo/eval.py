from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from lane_seg.data.dataset import SDLaneDataset
from lane_seg.evaluation.mask_metrics import dice_iou_prf_from_masks
from lane_seg.yolo.predict import load_yolo_model, yolo_predict_union_mask


@dataclass
class YoloEvalResult:
    dice: float
    iou: float
    precision: float
    recall: float
    f1: float


def eval_yolo_seg(
    cfg: Dict[str, Any],
    sdlane_root: Path,
    list_file: Path,
    split_dir: str,
    weights: str,
    threshold: float = 0.5,
    save_preview_dir: Optional[Path] = None,
    max_preview: int = 0,
) -> YoloEvalResult:
    """Evaluate YOLO-seg by converting predictions to a binary semantic mask.

    - Uses union of all predicted instance masks.
    - Compares with ground-truth lane semantic mask.
    - Optionally saves preview overlays.
    """
    ycfg = cfg.get("yolo", {}) or {}

    model = load_yolo_model(weights)

    # We do not use albumentations here; evaluate in original resolution
    ds = SDLaneDataset(
        sdlane_root=Path(sdlane_root),
        list_file=Path(list_file),
        cfg=cfg,
        transforms=None,
        split_dir=split_dir,
    )

    dice_s, iou_s, p_s, r_s, f1_s = [], [], [], [], []

    if save_preview_dir is not None:
        save_preview_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(ds)), desc="eval_yolo", total=len(ds)):
        # Load original image+mask
        scene, frame = ds._parse_item(ds.items[i])
        base = Path(sdlane_root) if split_dir == "" else (Path(sdlane_root) / split_dir)

        # image path
        img_path = None
        for ext in ds.image_exts:
            pth = base / "images" / scene / f"{frame}.{ext}"
            if pth.exists():
                img_path = pth
                break
        if img_path is None:
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]

        # ground-truth mask
        lbl_json = base / "labels" / scene / f"{frame}.json"
        mask_path = base / ds.mask_subdir / scene / f"{frame}.{ds.mask_ext}"

        if bool(cfg.get("data", {}).get("use_precomputed_masks", False)):
            try:
                gt01 = ds._load_precomputed_mask(mask_path, h, w)
            except Exception:
                if not bool(cfg.get("data", {}).get("fallback_to_json", True)):
                    raise
                gt01 = ds._label_to_mask(lbl_json, h, w)
        else:
            gt01 = ds._label_to_mask(lbl_json, h, w)

        pred = yolo_predict_union_mask(
            model,
            img_bgr,
            conf=float(ycfg.get("conf", 0.25)),
            iou=float(ycfg.get("iou", 0.7)),
            classes=ycfg.get("classes"),
            imgsz=ycfg.get("imgsz"),
        )
        pr01 = pred.binary_mask

        # optional thresholding hook (currently binary already)
        pr01 = (pr01 > 0).astype(np.uint8)

        d, j, pp, rr, ff = dice_iou_prf_from_masks(pr01, gt01)
        dice_s.append(d)
        iou_s.append(j)
        p_s.append(pp)
        r_s.append(rr)
        f1_s.append(ff)

        if save_preview_dir is not None and max_preview > 0 and i < max_preview:
            from lane_seg.yolo.predict import overlay_mask_on_bgr

            ov = overlay_mask_on_bgr(img_bgr, pr01)
            out_path = save_preview_dir / f"{i:05d}_{scene}_{frame}.jpg"
            cv2.imwrite(str(out_path), ov)

    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    return YoloEvalResult(
        dice=mean(dice_s),
        iou=mean(iou_s),
        precision=mean(p_s),
        recall=mean(r_s),
        f1=mean(f1_s),
    )
