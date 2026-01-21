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

    # paths.* (keep dotted override compatibility)
    p.add_argument("--paths.sdlane_root", dest="paths_sdlane_root")
    p.add_argument("--paths.out_dir", dest="paths_out_dir")
    p.add_argument("--paths.run_name", dest="paths_run_name")

    # yolo.* (IMPORTANT: use dest with underscores to avoid Namespace dot-attr issues)
    p.add_argument("--yolo.weights", dest="yolo_weights")
    p.add_argument("--yolo.imgsz", dest="yolo_imgsz", type=int)  # e.g. 640
    p.add_argument("--yolo.epochs", dest="yolo_epochs", type=int)
    p.add_argument("--yolo.batch", dest="yolo_batch", type=int)
    p.add_argument("--yolo.device", dest="yolo_device")
    p.add_argument("--yolo.conf", dest="yolo_conf", type=float)
    p.add_argument("--yolo.iou", dest="yolo_iou", type=float)
    return p


def _to_dotted_overrides(args_ns: argparse.Namespace) -> dict:
    """Convert underscore args into dotted keys expected by apply_overrides()."""
    a = vars(args_ns)
    overrides = {}

    # paths.*
    if a.get("paths_sdlane_root") is not None:
        overrides["paths.sdlane_root"] = a["paths_sdlane_root"]
    if a.get("paths_out_dir") is not None:
        overrides["paths.out_dir"] = a["paths_out_dir"]
    if a.get("paths_run_name") is not None:
        overrides["paths.run_name"] = a["paths_run_name"]

    # yolo.*
    if a.get("yolo_weights") is not None:
        overrides["yolo.weights"] = a["yolo_weights"]
    if a.get("yolo_imgsz") is not None:
        overrides["yolo.imgsz"] = a["yolo_imgsz"]
    if a.get("yolo_epochs") is not None:
        overrides["yolo.epochs"] = a["yolo_epochs"]
    if a.get("yolo_batch") is not None:
        overrides["yolo.batch"] = a["yolo_batch"]
    if a.get("yolo_device") is not None:
        overrides["yolo.device"] = a["yolo_device"]
    if a.get("yolo_conf") is not None:
        overrides["yolo.conf"] = a["yolo_conf"]
    if a.get("yolo_iou") is not None:
        overrides["yolo.iou"] = a["yolo_iou"]

    return overrides


def main():
    args = build_parser().parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, _to_dotted_overrides(args))

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
