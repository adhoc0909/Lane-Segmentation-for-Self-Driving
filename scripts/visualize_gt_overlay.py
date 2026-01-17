from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def imgpath_to_labelpath(img_path: Path, root: Path) -> Path:
    """
    train/images/scene/frame.jpg
    -> train/labels/scene/frame.json
    """
    rel = img_path.relative_to(root)
    parts = rel.parts

    # train/images/scene/frame.jpg
    if "images" not in parts:
        raise ValueError(f"'images' not in path: {img_path}")

    idx = parts.index("images")
    label_parts = list(parts)
    label_parts[idx] = "labels"
    label_parts[-1] = Path(label_parts[-1]).with_suffix(".json").name

    return root.joinpath(*label_parts)


def draw_geometry(mask: np.ndarray, geometry: list, thickness: int = 6):
    for lane in geometry:
        if len(lane) < 2:
            continue
        pts = np.asarray(lane, dtype=np.float32)

        pts[:, 0] = np.clip(pts[:, 0], 0, mask.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, mask.shape[0] - 1)

        pts = pts.astype(np.int32)
        cv2.polylines(mask, [pts], False, 1, thickness)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdlane_root", type=str, required=True)
    ap.add_argument("--img_relpath", type=str, required=True)
    ap.add_argument("--thickness", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(args.sdlane_root)
    img_path = root / args.img_relpath
    lbl_path = imgpath_to_labelpath(img_path, root)

    if not img_path.exists():
        raise FileNotFoundError(img_path)
    if not lbl_path.exists():
        raise FileNotFoundError(lbl_path)

    # ---------- load image ----------
    img_bgr = cv2.imread(str(img_path))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # ---------- load label ----------
    with open(lbl_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    geometry = data.get("geometry", [])
    if not geometry:
        raise ValueError("No geometry found in label JSON")

    # ---------- draw mask ----------
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_geometry(mask, geometry, thickness=args.thickness)

    # ---------- overlay ----------
    overlay = img.copy()
    overlay[mask > 0] = [255, 0, 0]  # red lanes

    vis = cv2.addWeighted(overlay, args.alpha, img, 1 - args.alpha, 0)

    # ---------- show ----------
    plt.figure(figsize=(10, 6))
    plt.imshow(vis)
    plt.title(f"GT Overlay\n{args.img_relpath}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
