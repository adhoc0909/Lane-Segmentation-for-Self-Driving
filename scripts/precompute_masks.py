from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_item(item: str) -> tuple[str, str]:
    item = item.strip().replace("\\", "/")
    if item.lower().endswith(".jpg") or item.lower().endswith(".png"):
        item = item.rsplit(".", 1)[0]
    parts = [p for p in item.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Unknown list format: {item}")
    return parts[-2], parts[-1]


def json_to_mask(label_path: Path, h: int, w: int, thickness: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ✅ SDLane format: {"geometry": [ [[x,y],...], [[x,y],...], ... ], "idx": [...] }
    lanes = data.get("geometry")
    if lanes is None:
        # fallback (older formats)
        lanes = data.get("lanes") or data.get("Lane") or data.get("lane") or []

    if not isinstance(lanes, list):
        return mask

    for lane in lanes:
        if not isinstance(lane, list) or len(lane) < 2:
            continue
        pts = np.asarray(lane, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            continue

        # clip to image bounds (some coords can be <0 or >W/H)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts = pts.astype(np.int32)

        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)

    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdlane_root", type=str, required=True, help="SDLane root folder (has images/ and labels/)")
    ap.add_argument("--list_file", type=str, required=True, help="split file (e.g., run_dir/splits/train.txt)")
    ap.add_argument("--out_subdir", type=str, default="masks", help="output folder name under sdlane_root")
    ap.add_argument("--img_ext", type=str, default="jpg", choices=["jpg", "png"], help="image extension in images/")
    ap.add_argument("--thickness", type=int, default=6, help="lane thickness (pixels)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fail_on_empty", action="store_true", help="raise error if any empty mask occurs")
    args = ap.parse_args()

    root = Path(args.sdlane_root)
    list_file = Path(args.list_file)
    out_root = root / args.out_subdir

    if not list_file.exists():
        raise FileNotFoundError(f"list_file not found: {list_file}")
    if not (root / "images").exists():
        raise FileNotFoundError(f"images/ not found under: {root}")
    if not (root / "labels").exists():
        raise FileNotFoundError(f"labels/ not found under: {root}")

    out_root.mkdir(parents=True, exist_ok=True)

    items = [l.strip() for l in list_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    n_total = len(items)
    n_ok = 0
    n_skip = 0
    n_empty = 0
    n_missing = 0

    for i, item in enumerate(items, 1):
        scene, frame = parse_item(item)

        img_path = root / "images" / scene / f"{frame}.{args.img_ext}"
        lbl_path = root / "labels" / scene / f"{frame}.json"
        out_dir = out_root / scene
        out_path = out_dir / f"{frame}.png"

        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        if not img_path.exists() or not lbl_path.exists():
            n_missing += 1
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            n_missing += 1
            continue
        h, w = img_bgr.shape[:2]

        mask = json_to_mask(lbl_path, h, w, thickness=args.thickness)
        if mask.sum() == 0:
            n_empty += 1
            if args.fail_on_empty:
                raise ValueError(f"Empty mask: {lbl_path}")

        out_dir.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), mask)
        if ok:
            n_ok += 1

        if i % 200 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] ok={n_ok} skip={n_skip} missing={n_missing} empty={n_empty} -> {out_root}"
            )

    print("\nDone.")
    print(f"total={n_total}, ok={n_ok}, skip={n_skip}, missing={n_missing}, empty={n_empty}")
    print(f"saved_dir: {out_root}")


if __name__ == "__main__":
    main()
