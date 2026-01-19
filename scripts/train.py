import argparse, os
import torch
from pathlib import Path

from _bootstrap import *  # noqa

from lane_seg.utils.config import load_yaml, apply_overrides, dump_yaml
from lane_seg.utils.seed import set_seed
from lane_seg.utils.paths import resolve_sdlane_root, make_run_dir
from lane_seg.utils.logging import CSVLogger
from lane_seg.utils.builders import build_loaders
from lane_seg.models.factory import build_model
from lane_seg.models.losses import build_loss
from lane_seg.engine.loops import train_one_epoch, validate
from lane_seg.engine.checkpoint import save_checkpoint


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--paths.sdlane_root")
    p.add_argument("--paths.out_dir")
    p.add_argument("--paths.run_name")
    p.add_argument("--data.val_ratio", type=float)
    p.add_argument("--data.num_workers", type=int)
    p.add_argument("--data.image_size", nargs=2, type=int)  # W H
    p.add_argument("--model.arch")
    p.add_argument("--model.encoder")
    p.add_argument("--model.pretrained")
    p.add_argument("--train.device")
    p.add_argument("--train.batch_size", type=int)
    p.add_argument("--train.epochs", type=int)
    p.add_argument("--train.lr", type=float)
    p.add_argument("--train.weight_decay", type=float)
    p.add_argument("--train.amp")
    p.add_argument("--train.grad_clip_norm", type=float)
    p.add_argument("--loss.name")
    p.add_argument("--eval.threshold", type=float)
    return p


def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def main():
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, vars(args))

    if cfg["paths"].get("sdlane_root") is None:
        cfg["paths"]["sdlane_root"] = os.environ.get("SDLANE_ROOT")

    # normalize bool-like overrides if provided as strings
    if isinstance(cfg["model"].get("pretrained"), str):
        cfg["model"]["pretrained"] = str2bool(cfg["model"]["pretrained"])
    if isinstance(cfg["train"].get("amp"), str):
        cfg["train"]["amp"] = str2bool(cfg["train"]["amp"])

    set_seed(int(cfg["project"]["seed"]))

    sdlane_root = resolve_sdlane_root(cfg)
    run_dir = make_run_dir(cfg)
    dump_yaml(run_dir / "resolved_config.yaml", cfg)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, _ = build_loaders(cfg, sdlane_root, run_dir)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    sch_cfg = cfg["train"].get("scheduler", {"name": "none"})
    sch_name = (sch_cfg.get("name") or "none").lower()
    scheduler = None
    if sch_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(sch_cfg.get("t_max", cfg["train"]["epochs"]))
        )
    elif sch_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=int(sch_cfg.get("step_size", 10)), gamma=float(sch_cfg.get("gamma", 0.1))
        )

    logger = CSVLogger(
        run_dir / "metrics.csv",
        fieldnames=["epoch", "train_loss", "val_loss", "dice", "iou", "precision", "recall", "f1", "lr"],
    )

    best = -1e9
    metric_name = cfg["train"]["checkpoint"]["metric"]
    mode = cfg["train"]["checkpoint"]["mode"]
    thr = float(cfg["eval"]["threshold"])

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            loss_fn,
            device,
            amp=bool(cfg["train"]["amp"]),
            grad_clip_norm=float(cfg["train"]["grad_clip_norm"]),
        )

        metrics = validate(model, val_loader, loss_fn, device, thr=thr)
        lr = opt.param_groups[0]["lr"]

        logger.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": metrics["val_loss"],
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "lr": lr,
            }
        )

        save_every = int(cfg["train"]["checkpoint"].get("save_every", 1))
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, opt, epoch, metrics, cfg)

        current = metrics[metric_name]
        improved = current > best if mode == "max" else current < best
        if improved:
            best = current
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, opt, epoch, metrics, cfg)

        if scheduler is not None:
            scheduler.step()

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"dice={metrics['dice']:.4f} "
            f"iou={metrics['iou']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"best_{metric_name}={best:.4f}"
        )

    print(f"✅ Done. outputs: {run_dir}")


if __name__ == "__main__":
    main()
