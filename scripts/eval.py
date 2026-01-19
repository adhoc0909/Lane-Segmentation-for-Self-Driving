import argparse, os
from pathlib import Path
import torch

from _bootstrap import *  # noqa

from lane_seg.utils.config import load_yaml, apply_overrides
from lane_seg.utils.paths import resolve_sdlane_root, make_run_dir
from lane_seg.utils.builders import build_loaders
from lane_seg.models.factory import build_model
from lane_seg.models.losses import build_loss
from lane_seg.engine.loops import validate
from lane_seg.engine.checkpoint import load_weights


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--paths.sdlane_root")
    p.add_argument("--paths.out_dir")
    p.add_argument("--paths.run_name")
    p.add_argument("--infer.weights")
    p.add_argument("--eval.threshold", type=float)
    return p


def main():
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, vars(args))

    if cfg["paths"].get("sdlane_root") is None:
        cfg["paths"]["sdlane_root"] = os.environ.get("SDLANE_ROOT")

    sdlane_root = resolve_sdlane_root(cfg)
    run_dir = make_run_dir(cfg)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # ✅ test=True일 때 build_loaders는 (test_loader, test_txt)를 반환
    test_loader, _ = build_loaders(cfg, sdlane_root, run_dir, test=True)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)

    weights = cfg["infer"].get("weights")
    if weights is None:
        weights = run_dir / "checkpoints" / "best.pt"
    load_weights(Path(weights), model)

    thr = float(cfg["eval"]["threshold"])
    metrics = validate(model, test_loader, loss_fn, device, thr=thr)
    print(f"✅ Eval metrics @thr={thr}: {metrics}")


if __name__ == "__main__":
    main()
