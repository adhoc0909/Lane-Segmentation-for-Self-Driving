import argparse
import os
from pathlib import Path

from _bootstrap import *  # noqa

from lane_seg.utils.config import load_yaml, apply_overrides, dump_yaml
from lane_seg.utils.seed import set_seed
from lane_seg.utils.paths import resolve_sdlane_root, make_run_dir
from lane_seg.yolo.eval import eval_yolo_seg


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--paths.sdlane_root")
    p.add_argument("--paths.out_dir")
    p.add_argument("--paths.run_name")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--save_preview", action="store_true")
    p.add_argument("--max_preview", type=int, default=30)
    return p


def main():
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, vars(args))

    if cfg["paths"].get("sdlane_root") is None:
        cfg["paths"]["sdlane_root"] = os.environ.get("SDLANE_ROOT")

    set_seed(int(cfg["project"]["seed"]))

    sdlane_root = resolve_sdlane_root(cfg)
    run_dir = make_run_dir(cfg)

    # pick list/split_dir
    if args.split == "test":
        split_dir = cfg["data"]["splits"]["test"]["dir"]
        list_file = Path(sdlane_root) / split_dir / cfg["data"]["splits"]["test"]["list"]
    else:
        # val list is generated in run_dir/splits by existing split util
        from lane_seg.data.split import make_splits

        train_dir = cfg["data"]["splits"]["train"]["dir"]
        train_list = Path(sdlane_root) / train_dir / cfg["data"]["splits"]["train"]["list"]
        tr_txt, va_txt = make_splits(train_list, run_dir / "splits", float(cfg["data"].get("val_ratio", 0.2)), int(cfg["project"].get("seed", 42)))
        split_dir = train_dir
        list_file = va_txt

    preview_dir = (run_dir / "yolo_previews") if args.save_preview else None

    res = eval_yolo_seg(
        cfg,
        sdlane_root=Path(sdlane_root),
        list_file=Path(list_file),
        split_dir=split_dir,
        weights=args.weights,
        threshold=float(cfg.get("eval", {}).get("threshold", 0.5)),
        save_preview_dir=preview_dir,
        max_preview=int(args.max_preview),
    )

    print("\n[YOLO EVAL]")
    print(f"dice={res.dice:.4f} iou={res.iou:.4f} precision={res.precision:.4f} recall={res.recall:.4f} f1={res.f1:.4f}")


if __name__ == "__main__":
    main()
