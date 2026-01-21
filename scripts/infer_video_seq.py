from __future__ import annotations

"""Video inference for sequence-based (recurrent) models.

This is intentionally separate from scripts/infer_video.py to avoid
changing existing behavior.

Supports:
  - ONNXRuntime (.onnx) exported from this repo's recurrent models
  - PyTorch weights (.pt/.pth) using build_model(cfg)

Example:
python scripts/infer_video_seq.py \
  --config configs/unet_gru_seq3.yaml \
  --weights outputs/.../checkpoints/best.onnx \
  --video /path/to/sample.mp4 \
  --out /path/to/out_seq3.mp4 \
  --seq_len 3 --thr 0.5 --alpha 0.35
"""

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


# -----------------------------
# bootstrap
# -----------------------------
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lane_seg.utils.config import load_yaml, apply_overrides  # noqa: E402
from lane_seg.models.factory import build_model              # noqa: E402
from lane_seg.engine.checkpoint import load_weights          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--thr", type=float, default=None)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--seq_len", type=int, default=None, help="Override sequence length (T).")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--resize", nargs=2, type=int, default=None, metavar=("W", "H"))
    p.add_argument("--print_every", type=int, default=120)
    return p.parse_args()


def overlay_color(bgr: np.ndarray, mask01: np.ndarray, alpha: float, color_bgr=(255, 255, 0)) -> np.ndarray:
    out = bgr.copy()
    m = mask01.astype(bool)
    if not np.any(m):
        return out
    overlay = np.zeros_like(out)
    overlay[:] = color_bgr
    out[m] = (alpha * overlay[m] + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def preprocess_rgb_to_nchw(rgb: np.ndarray) -> np.ndarray:
    x = (rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)) / 255.0
    return x


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, vars(args))

    thr = args.thr if args.thr is not None else float(cfg.get("eval", {}).get("threshold", 0.5))

    device_str = args.device or cfg.get("train", {}).get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in str(device_str) else "cpu")

    weights_path = Path(args.weights)
    is_onnx = weights_path.suffix.lower() == ".onnx"

    ort_sess = None
    ort_in_name = None

    if is_onnx:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "ONNX inference requires onnxruntime. Install one of:\n"
                "  pip install onnxruntime  (CPU)\n"
                "  pip install onnxruntime-gpu  (CUDA)\n"
            ) from e

        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        ort_sess = ort.InferenceSession(str(weights_path), providers=providers)
        ort_in_name = ort_sess.get_inputs()[0].name
        print("ONNX inputs:", [(i.name, i.shape, i.type) for i in ort_sess.get_inputs()])
        print("ONNX outputs:", [(o.name, o.shape, o.type) for o in ort_sess.get_outputs()])
    else:
        model = build_model(cfg).to(device).eval()
        load_weights(weights_path, model)

    # model input size
    img_size = cfg.get("data", {}).get("image_size", None)
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        net_w, net_h = int(img_size[0]), int(img_size[1])
    else:
        net_w, net_h = None, None

    # sequence length
    seq_len = args.seq_len
    if seq_len is None:
        seq_len = int(cfg.get("data", {}).get("sequence_len", 3))
    seq_len = max(1, int(seq_len))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.resize is not None:
        out_w, out_h = int(args.resize[0]), int(args.resize[1])
    else:
        out_w, out_h = src_w, src_h
    if net_w is None or net_h is None:
        net_w, net_h = out_w, out_h

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (out_w, out_h))

    q: deque[np.ndarray] = deque(maxlen=seq_len)
    frame_i = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_i += 1
        if args.max_frames and frame_i > args.max_frames:
            break

        if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        inp = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        q.append(rgb)

        # pad sequence initially by repeating first frame
        if len(q) < seq_len:
            while len(q) < seq_len:
                q.appendleft(q[0])

        # build [1,T,C,H,W]
        xs = [preprocess_rgb_to_nchw(im)[0] for im in list(q)]  # each: [C,H,W]
        x_np = np.stack(xs, axis=0)[None, ...].astype(np.float32)  # [1,T,C,H,W]

        if is_onnx:
            assert ort_sess is not None and ort_in_name is not None
            out = ort_sess.run(None, {ort_in_name: x_np})[0]
            # Expect [1,1,H,W] or [1,H,W]
            if out.ndim == 3:
                out = out[:, None, :, :]
            logits = out
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
            mask01 = (prob[0, 0] > float(thr)).astype(np.uint8)
        else:
            x_t = torch.from_numpy(x_np).to(device)
            logits = model(x_t)  # [B,1,H,W]
            prob = torch.sigmoid(logits)
            mask01 = (prob[0, 0] > float(thr)).to(torch.uint8).cpu().numpy()

        # resize mask to output size
        if mask01.shape[:2] != (out_h, out_w):
            mask01 = cv2.resize(mask01, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        vis = overlay_color(frame, mask01, alpha=float(args.alpha), color_bgr=(255, 255, 0))  # cyan-ish
        vw.write(vis)

        if args.print_every > 0 and (frame_i % int(args.print_every) == 0):
            dt = time.time() - t0
            fps_now = frame_i / max(dt, 1e-6)
            print(f"[{frame_i}] fps={fps_now:.1f}")

    cap.release()
    vw.release()
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
