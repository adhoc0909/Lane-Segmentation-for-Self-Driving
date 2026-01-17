import argparse, os
from pathlib import Path
import torch

from scripts._bootstrap import *  # noqa

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
    _, val_loader, _, _ = build_loaders(cfg, sdlane_root, run_dir)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)

    weights = cfg["infer"].get("weights")
    if weights is None:
        weights = run_dir / "checkpoints" / "best.pt"
    load_weights(Path(weights), model)

    metrics = validate(model, val_loader, loss_fn, device, thr=float(cfg["eval"]["threshold"]))
    print(f"✅ Eval metrics: {metrics}")

if __name__ == "__main__":
    main()
