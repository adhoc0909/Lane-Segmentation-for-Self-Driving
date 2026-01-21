from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


# =========================
# Utils
# =========================
def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


# =========================
# ONNX
# =========================
class OnnxSegModel:
    def __init__(self, weight: Path):
        if ort is None:
            raise RuntimeError("onnxruntime required")
        self.sess = ort.InferenceSession(str(weight), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.h = int(inp.shape[2])
        self.w = int(inp.shape[3])
        self.out_name = self.sess.get_outputs()[0].name

    def infer(self, frame):
        img = cv2.resize(frame, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        x = img.transpose(2, 0, 1)[None]
        out = self.sess.run([self.out_name], {self.in_name: x})[0]
        while out.ndim > 2:
            out = out[0]
        return out


# =========================
# Postprocess (MINIMAL)
# =========================
def remove_blob_and_connect(mask01: np.ndarray) -> np.ndarray:
    """
    mask01: HxW (0/1)
    return: HxW (0/1)
    """

    h, w = mask01.shape
    bin255 = (mask01 > 0).astype(np.uint8) * 255

    # -------------------------
    # 1. Connected Components (blob 제거)
    # -------------------------
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin255, connectivity=8)
    clean = np.zeros_like(bin255)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # 🔴 blob 제거 기준 (보수적)
        if area > 5000:       # 너무 큰 덩어리
            continue
        if height < 20:       # 너무 짧음
            continue

        clean[labels == i] = 255

    # -------------------------
    # 2. Skeletonize (가늘게)
    # -------------------------
    skel = np.zeros_like(clean)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = clean.copy()

    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        subset = cv2.subtract(temp, opened)
        skel = cv2.bitwise_or(skel, subset)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # -------------------------
    # 3. 점선 연결 (아주 약하게)
    # -------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skel = cv2.dilate(skel, kernel, iterations=1)

    return (skel > 0).astype(np.float32)


# =========================
# Overlay
# =========================
def overlay(frame, mask01):
    color = (255, 255, 0)
    out = frame.copy()
    out[mask01 > 0] = (
        0.7 * out[mask01 > 0] + 0.3 * np.array(color)
    ).astype(np.uint8)
    return out


# =========================
# Writer (ffmpeg fallback)
# =========================
class Writer:
    def __init__(self, out_path, fps, size: Tuple[int, int]):
        ensure_parent(out_path)
        if shutil.which("ffmpeg"):
            w, h = size
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "26",
                "-pix_fmt", "yuv420p",
                str(out_path)
            ]
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.use_ffmpeg = True
        else:
            self.vw = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                size
            )
            self.use_ffmpeg = False

    def write(self, frame):
        if self.use_ffmpeg:
            self.p.stdin.write(frame.tobytes())
        else:
            self.vw.write(frame)

    def close(self):
        if self.use_ffmpeg:
            self.p.stdin.close()
            self.p.wait()
        else:
            self.vw.release()


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    model = OnnxSegModel(Path(args.weights))
    cap = cv2.VideoCapture(args.video)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) / args.stride

    out_w = int(src_w * args.scale)
    out_h = int(src_h * args.scale)

    writer = Writer(Path(args.out), fps, (out_w, out_h))

    idx = 0

    if args.preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        idx += 1
        if idx % args.stride != 0:
            continue

        logits = model.infer(frame)
        prob = sigmoid(logits)
        mask = (prob > args.thr).astype(np.float32)

        mask = remove_blob_and_connect(mask)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        vis = overlay(frame, mask)
        vis = cv2.resize(vis, (out_w, out_h))

        writer.write(vis)

        if args.preview:
            cv2.imshow("preview", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.close()
    if args.preview:
        cv2.destroyAllWindows()

    print(f"[DONE] saved: {args.out}")


if __name__ == "__main__":
    main()
