"""Unified entrypoint for both U-Net (semantic segmentation) and YOLO11-seg (instance -> union mask).

Examples
--------
# U-Net train (existing codepath)
python scripts/run.py train --backend unet --config configs/default.yaml

# YOLO train
python scripts/run.py train --backend yolo --config configs/default.yaml --yolo.weights yolo11n-seg.pt

# U-Net eval
python scripts/run.py eval --backend unet --config configs/default.yaml --weights outputs/exp001/checkpoints/best.pt

# YOLO eval
python scripts/run.py eval --backend yolo --config configs/default.yaml --weights outputs/exp001/yolo/weights/best.pt

# Video inference
python scripts/run.py infer_video --backend yolo --weights outputs/exp001/yolo/weights/best.pt --video input.mp4 --out out.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_py(script_rel: str, argv: list[str]) -> int:
    script = _repo_root() / "scripts" / script_rel
    if not script.exists():
        raise FileNotFoundError(script)
    cmd = [sys.executable, str(script)] + argv
    return subprocess.call(cmd)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "eval", "infer_video"], help="What to run")
    p.add_argument("--backend", choices=["unet", "yolo"], required=True)
    p.add_argument("--config", default="configs/default.yaml")
    # Generic
    p.add_argument("--weights", default=None)
    p.add_argument("--video", default=None)
    p.add_argument("--out", default=None)

    # passthrough options (keep flexible)
    p.add_argument("args", nargs=argparse.REMAINDER, help="Extra args passed to underlying script")
    return p


def main():
    args = build_parser().parse_args()

    # Clean '--' separator if user uses it
    extra = args.args
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.command == "train":
        if args.backend == "unet":
            # existing script
            return _run_py("train.py", ["--config", args.config] + extra)
        return _run_py("train_yolo.py", ["--config", args.config] + extra)

    if args.command == "eval":
        if args.weights is None:
            raise SystemExit("--weights is required for eval")
        if args.backend == "unet":
            return _run_py("eval.py", ["--config", args.config, "--weights", args.weights] + extra)
        return _run_py("eval_yolo.py", ["--config", args.config, "--weights", args.weights] + extra)

    if args.command == "infer_video":
        if args.weights is None or args.video is None:
            raise SystemExit("--weights and --video are required for infer_video")
        if args.backend == "unet":
            argv = ["--config", args.config, "--weights", args.weights, "--video", args.video]
            if args.out is not None:
                argv += ["--out", args.out]
            return _run_py("infer_video.py", argv + extra)

        argv = ["--weights", args.weights, "--video", args.video]
        if args.out is not None:
            argv += ["--out", args.out]
        return _run_py("infer_video_yolo.py", argv + extra)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
