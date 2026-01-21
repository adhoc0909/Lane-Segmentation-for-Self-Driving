# python scripts/infer_video.py \
#   --weights /Users/leehw/Documents/likelion/2nd_project/Lane-Segmentation-for-Self-Driving/outputs/exp001/checkpoints/unet_60epoch_best.onnx \
#   --video /Volumes/T7/블박/sample2.mp4 \
#   --out /Volumes/T7/블박/sample2_unet_60epoch_best.mp4 \
#   --thr 0.5 \
#   --resize 640 400


# python scripts/infer_video.py \
#   --weights /Volumes/T7/블박/outputs/yoloseg_30epoch_best_1024_1024.onnx \
#   --video /Volumes/T7/블박/sample2.mp4 \
#   --out /Volumes/T7/블박/sample2_yoloseg_30epoch_best_1024_1024.mp4 \
#   --thr 0.5 \
#   --resize 1024 1024



from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

onnx_dynamic_hw = False

# -----------------------------
# bootstrap: allow "python scripts/infer_video.py" from repo root
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
    p.add_argument("--out", type=str, default="./_infer_out.mp4")
    p.add_argument("--device", type=str, default=None)  # "cuda" or "cpu"
    p.add_argument("--thr", type=float, default=None)
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--max_frames", type=int, default=0)  # 0 = all
    p.add_argument("--resize", nargs=2, type=int, default=None, metavar=("W", "H"))
    p.add_argument("--save_mask", action="store_true")   # also save mask video
    return p.parse_args()

def pad_to_stride(img: np.ndarray, stride: int = 32) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Pad H,W to be divisible by stride.
    Returns padded_img and pads (top, bottom, left, right).
    """
    h, w = img.shape[:2]
    pad_h = (stride - (h % stride)) % stride
    pad_w = (stride - (w % stride)) % stride

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if pad_h == 0 and pad_w == 0:
        return img, (0, 0, 0, 0)

    padded = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded, (top, bottom, left, right)


def unpad(img: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pads
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return img
    h, w = img.shape[:2]
    return img[top:h - bottom, left:w - right]



def overlay_red(bgr: np.ndarray, mask01: np.ndarray, alpha: float) -> np.ndarray:
    """mask01: HxW (0/1)"""
    out = bgr.copy()
    m = mask01.astype(bool)
    if not np.any(m):
        return out
    red = np.zeros_like(out)
    red[..., 2] = 255
    out[m] = (alpha * red[m] + (1 - alpha) * out[m]).astype(np.uint8)
    return out


@torch.no_grad()
def main():
    args = parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, vars(args))  # allow --thr etc. if you want to extend
    thr = args.thr if args.thr is not None else float(cfg.get("eval", {}).get("threshold", 0.5))

    device = args.device
    if device is None:
        device = cfg.get("train", {}).get("device", "cuda")
    device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")

    weights_path = Path(args.weights)
    is_onnx = weights_path.suffix.lower() == ".onnx"

    ort_sess = None
    ort_input_name = None

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
        print("ONNX inputs:", [(i.name, i.shape, i.type) for i in ort_sess.get_inputs()])
        print("ONNX outputs:", [(o.name, o.shape, o.type) for o in ort_sess.get_outputs()])
        in_shape = ort_sess.get_inputs()[0].shape  # e.g. [1, 3, 400, 640] or [1,3,'height','width']
        # Only allow padding when H/W are dynamic (not fixed ints)
        onnx_dynamic_hw = not (isinstance(in_shape[2], int) and isinstance(in_shape[3], int))

        ort_input_name = ort_sess.get_inputs()[0].name
    else:
        model = build_model(cfg).to(device).eval()
        load_weights(weights_path, model)

    video_path = str(args.video)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # output video settings
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

    # model input size (from cfg or fallback to frame size)
    # If your transforms already resize in dataset, inference should also match.
    # We'll use cfg.data.image_size if present.
    img_size = cfg.get("data", {}).get("image_size", None)
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        net_w, net_h = int(img_size[0]), int(img_size[1])
    else:
        net_w, net_h = out_w, out_h

    # NOTE: This assumes your model expects normalized RGB in [0,1].
    # If you used other normalization in training transforms, replicate it here.
    frame_i = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_i += 1
        if args.max_frames and frame_i > args.max_frames:
            break

        # resize for output
        if (out_w, out_h) != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # prepare model input
        inp = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        pads = (0, 0, 0, 0)
        if is_onnx and onnx_dynamic_hw:
            rgb, pads = pad_to_stride(rgb, stride=32)




        if is_onnx:
            # ONNXRuntime expects NCHW float32
            x_np = (rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)) / 255.0
            out = ort_sess.run(None, {ort_input_name: x_np})[0]
            # out: (1,1,H,W) or (1,H,W) or (1,H,W,1)
            if out.ndim == 4 and out.shape[-1] == 1:
                out = np.transpose(out, (0, 3, 1, 2))
            if out.ndim == 3:
                out = out[:, None, :, :]
            out2d = out[0, 0]
            # Heuristic: if already probability (0~1), don't apply sigmoid again
            if out2d.min() >= 0.0 and out2d.max() <= 1.0:
                prob2d = out2d
            else:
                prob2d = 1.0 / (1.0 + np.exp(-out2d))
            pred = (prob2d > thr).astype(np.uint8)
            pred = unpad(pred, pads)
        else:
            x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0).to(device)

            logits = model(x)                     # [1,1,H,W] expected
            prob = torch.sigmoid(logits)[0, 0]    # [H,W]
            pred = (prob > thr).to(torch.uint8).cpu().numpy()  # 0/1

        # resize mask back to output size
        if (net_w, net_h) != (out_w, out_h):
            pred = cv2.resize(pred, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        
        vis = overlay_red(frame, pred, alpha=args.alpha)

        # small HUD
        cv2.putText(
            vis,
            f"thr={thr:.2f} frame={frame_i}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"thr={thr:.2f} frame={frame_i}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        vw.write(vis)

        if mask_vw is not None:
            m = (pred * 255).astype(np.uint8)
            m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            mask_vw.write(m3)

        # progress
        if frame_i % 100 == 0:
            dt = time.time() - t0
            print(f"[{frame_i}] {frame_i/dt:.2f} fps (wall)")

    cap.release()
    vw.release()
    if mask_vw is not None:
        mask_vw.release()

    print(f"✅ saved: {out_path}")
    if args.save_mask:
        print(f"✅ saved mask video: {out_path.with_name(out_path.stem + '_mask.mp4')}")


if __name__ == "__main__":
    main()
