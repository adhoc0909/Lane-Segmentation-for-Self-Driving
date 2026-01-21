import argparse
import os
from pathlib import Path

from _bootstrap import *  # noqa

from lane_seg.utils.config import load_yaml, apply_overrides, dump_yaml
from lane_seg.utils.seed import set_seed
from lane_seg.utils.paths import resolve_sdlane_root, make_run_dir
from lane_seg.yolo.train import train_yolo_seg


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--paths.sdlane_root")
    p.add_argument("--paths.out_dir")
    p.add_argument("--paths.run_name")
    # YOLO overrides
    p.add_argument("--yolo.weights")
    p.add_argument("--yolo.imgsz", type=int, nargs="+", default=[640, 640])
    p.add_argument("--yolo.epochs", type=int)
    p.add_argument("--yolo.batch", type=int)
    p.add_argument("--yolo.device")
    p.add_argument("--yolo.conf", type=float)
    p.add_argument("--yolo.iou", type=float)
    return p


def main():
    args = build_parser().parse_args()
    imgsz = args.yolo.imgsz  # 예: [640] 또는 [640, 400]

    if isinstance(imgsz, list):
        if len(imgsz) == 1:
            imgsz = imgsz[0]
        elif len(imgsz) == 2:
            imgsz = tuple(imgsz)
        else:
            raise ValueError("--yolo.imgsz must be 1 or 2 integers (e.g., 640 or 640 400)")

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, vars(args))

    if cfg["paths"].get("sdlane_root") is None:
        cfg["paths"]["sdlane_root"] = os.environ.get("SDLANE_ROOT")

    set_seed(int(cfg["project"]["seed"]))

    sdlane_root = resolve_sdlane_root(cfg)
    run_dir = make_run_dir(cfg)
    dump_yaml(run_dir / "resolved_config.yaml", cfg)

    res = train_yolo_seg(cfg, sdlane_root=sdlane_root, run_dir=run_dir)

    print("\n[YOLO TRAIN DONE]")
    print("run_dir:", res.run_dir)
    print("best:", res.best_weights)
    print("last:", res.last_weights)


if __name__ == "__main__":
    main()
