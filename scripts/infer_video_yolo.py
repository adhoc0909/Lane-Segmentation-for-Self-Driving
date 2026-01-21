from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2

from _bootstrap import *  # noqa

from lane_seg.yolo.predict import load_yolo_model, yolo_predict_union_mask, overlay_mask_on_bgr, save_mask_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="YOLO -seg .pt weights")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out", type=str, default="./_infer_yolo.mp4")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--resize", nargs=2, type=int, default=None, metavar=("W", "H"))
    p.add_argument("--save_mask", action="store_true", help="Also save a binary mask video")
    p.add_argument("--save_mask_dir", type=str, default=None, help="If set, also save per-frame mask PNGs")
    return p.parse_args()


def main():
    args = parse_args()

    model = load_yolo_model(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.resize is not None:
        out_w, out_h = int(args.resize[0]), int(args.resize[1])
    else:
        out_w, out_h = in_w, in_h

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    mask_vw = None
    if args.save_mask:
        mask_out = out_path.with_name(out_path.stem + "_mask.mp4")
        mask_vw = cv2.VideoWriter(str(mask_out), fourcc, fps, (out_w, out_h))

    mask_dir = Path(args.save_mask_dir) if args.save_mask_dir else None
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)

    frame_i = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_i += 1
        if args.max_frames and frame_i > args.max_frames:
            break

        if (out_w, out_h) != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        pred = yolo_predict_union_mask(
            model,
            frame,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=args.imgsz,
        )

        overlay = overlay_mask_on_bgr(frame, pred.binary_mask, alpha=float(args.alpha))
        vw.write(overlay)

        if mask_vw is not None:
            m = (pred.binary_mask.astype("uint8") * 255)
            m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            mask_vw.write(m3)

        if mask_dir is not None:
            save_mask_png(pred.binary_mask, mask_dir / f"{frame_i:06d}.png")

    cap.release()
    vw.release()
    if mask_vw is not None:
        mask_vw.release()

    dt = time.time() - t0
    fps_eff = frame_i / max(dt, 1e-9)
    print(f"[DONE] frames={frame_i} time={dt:.2f}s ({fps_eff:.2f} FPS) -> {out_path}")


if __name__ == "__main__":
    main()
