from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from lane_seg.data.dataset import SDLaneDataset
from lane_seg.data.split import make_splits
from lane_seg.utils.config import dump_yaml


@dataclass
class YoloSegDatasetPaths:
    root: Path
    data_yaml: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path


def _sha1_of_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def mask_to_polygons(mask01: np.ndarray, simplify_eps: float = 1.5, min_area: float = 16.0) -> List[np.ndarray]:
    """Convert binary mask (0/1) to list of polygons (Nx2 int arrays).

    YOLO segmentation expects polygon points in normalized xy coordinates.

    Notes:
      - We use external contours.
      - We simplify contours to reduce label size.
      - Small components are filtered by area.
    """
    m = (mask01.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[np.ndarray] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        if simplify_eps > 0:
            peri = float(cv2.arcLength(cnt, closed=True))
            eps = simplify_eps / 100.0 * peri
            approx = cv2.approxPolyDP(cnt, eps, closed=True)
            cnt_use = approx
        else:
            cnt_use = cnt

        pts = cnt_use.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        polys.append(pts)

    return polys


def polygon_to_yolo_line(poly_xy: np.ndarray, cls: int, w: int, h: int) -> str:
    """Build one YOLO-seg label line: <cls> x1 y1 x2 y2 ... normalized."""
    pts = poly_xy.astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    flat = pts.reshape(-1)
    vals = " ".join(f"{v:.6f}" for v in flat.tolist())
    return f"{cls} {vals}"


def _copy_or_link(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        try:
            dst.symlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def prepare_yolo_seg_dataset(cfg: dict, sdlane_root: Path, run_dir: Path) -> YoloSegDatasetPaths:
    """Prepare a YOLO segmentation dataset directory from SDLane-style masks.

    This converts the *semantic* lane mask into YOLO *instance segmentation* format
    by extracting polygons from the binary lane mask.

    Output structure (Ultralytics standard):
      <root>/train/images, <root>/train/labels
      <root>/val/images,   <root>/val/labels
      <root>/test/images,  <root>/test/labels
      <root>/data.yaml

    We store the dataset under run_dir/cache_yolo to keep experiments reproducible.
    """
    ycfg = cfg.get("yolo", {}) or {}
    link_files = bool(ycfg.get("link_files", True))

    # determine splits using existing split utility
    split_cache = run_dir / "splits"
    train_list = Path(sdlane_root) / cfg["data"]["splits"]["train"]["dir"] / cfg["data"]["splits"]["train"]["list"]
    test_list = Path(sdlane_root) / cfg["data"]["splits"]["test"]["dir"] / cfg["data"]["splits"]["test"]["list"]

    tr_txt, va_txt = make_splits(train_list, split_cache, float(cfg["data"].get("val_ratio", 0.2)), int(cfg["project"].get("seed", 42)))

    # cache directory key by important params
    key = _sha1_of_str(f"{sdlane_root}|{tr_txt.read_text()}|{va_txt.read_text()}|{test_list.read_text()}|{cfg['data'].get('use_precomputed_masks')}|{cfg['data'].get('mask_subdir')}|{cfg['data'].get('mask_ext')}|{cfg['data'].get('lane_thickness')}")
    out_root = run_dir / "cache_yolo" / f"yolo_seg_{key}"

    train_dir = out_root / "train"
    val_dir = out_root / "val"
    test_dir = out_root / "test"

    data_yaml = out_root / "data.yaml"

    if data_yaml.exists():
        return YoloSegDatasetPaths(out_root, data_yaml, train_dir, val_dir, test_dir)

    # Build dataset reader to generate masks consistently
    # We will use precomputed masks if configured.
    data_cfg = cfg.get("data", {}) or {}

    # Helper to write split
    def process_split(split_name: str, list_path: Path, split_root_dir_name: str, out_split_dir: Path) -> None:
        # SDLaneDataset expects list file and split_dir to resolve actual image/label paths.
        ds = SDLaneDataset(
            sdlane_root=Path(sdlane_root),
            list_file=list_path,
            cfg=cfg,
            transforms=None,
            split_dir=split_root_dir_name,
        )

        # For efficiency, we read images via dataset private method and build mask similarly.
        for item in ds.items:
            scene, frame = ds._parse_item(item)
            base = Path(sdlane_root) if split_root_dir_name == "" else (Path(sdlane_root) / split_root_dir_name)

            # find image file (keep original extension)
            img_path = None
            for ext in ds.image_exts:
                p = base / "images" / scene / f"{frame}.{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                continue

            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]

            lbl_json = base / "labels" / scene / f"{frame}.json"
            mask_path = base / ds.mask_subdir / scene / f"{frame}.{ds.mask_ext}"

            if bool(data_cfg.get("use_precomputed_masks", False)):
                try:
                    mask01 = ds._load_precomputed_mask(mask_path, h, w)
                except Exception:
                    if not bool(data_cfg.get("fallback_to_json", True)):
                        raise
                    mask01 = ds._label_to_mask(lbl_json, h, w)
            else:
                mask01 = ds._label_to_mask(lbl_json, h, w)

            # polygons
            polys = mask_to_polygons(
                mask01,
                simplify_eps=float(ycfg.get("poly_simplify_eps", 1.5)),
                min_area=float(ycfg.get("poly_min_area", 16.0)),
            )

            # output paths
            out_img = out_split_dir / "images" / scene / img_path.name
            out_lbl = out_split_dir / "labels" / scene / f"{frame}.txt"

            _copy_or_link(img_path, out_img, link=link_files)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)

            # Single-class: lane = 0
            if len(polys) == 0:
                out_lbl.write_text("", encoding="utf-8")
            else:
                lines = [polygon_to_yolo_line(p, cls=0, w=w, h=h) for p in polys]
                out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Create
    process_split("train", tr_txt, cfg["data"]["splits"]["train"]["dir"], train_dir)
    process_split("val", va_txt, cfg["data"]["splits"]["train"]["dir"], val_dir)
    process_split("test", test_list, cfg["data"]["splits"]["test"]["dir"], test_dir)

    # data.yaml (Ultralytics)
    ydata = {
        "path": str(out_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "lane"},
        "nc": 1,
    }
    dump_yaml(data_yaml, ydata)

    return YoloSegDatasetPaths(out_root, data_yaml, train_dir, val_dir, test_dir)
