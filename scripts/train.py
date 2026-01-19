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
    p.add_argument("--train.resume")  # ✅ resume ckpt path
    p.add_argument("--loss.name")
    p.add_argument("--eval.threshold", type=float)
    return p


def str2bool(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def load_checkpoint(ckpt_path: Path, model, optimizer=None, device="cpu"):
    """
    Robust checkpoint loader.
    Supports common keys:
      - model_state / model / state_dict
      - optimizer_state / optimizer
      - epoch
      - metrics (dict)
      - best (float)
      - bad_count (int)   (optional; for early stop continuity)
    """
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict):
        # model
        if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            model.load_state_dict(ckpt["model_state"], strict=True)
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            model.load_state_dict(ckpt["state_dict"], strict=True)
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            model.load_state_dict(ckpt["model"], strict=True)
        else:
            # some checkpoints directly are state_dict-like
            # try loading ckpt as state_dict
            try:
                model.load_state_dict(ckpt, strict=True)
            except Exception as e:
                raise RuntimeError(f"Unsupported checkpoint format for model. keys={list(ckpt.keys())}") from e

        # optimizer
        if optimizer is not None:
            opt_state = None
            if "optimizer_state" in ckpt and isinstance(ckpt["optimizer_state"], dict):
                opt_state = ckpt["optimizer_state"]
            elif "optimizer" in ckpt and isinstance(ckpt["optimizer"], dict):
                opt_state = ckpt["optimizer"]
            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception as e:
                    print(f"[warn] failed to load optimizer state: {e}")

        epoch = int(ckpt.get("epoch", 0) or 0)
        metrics = ckpt.get("metrics", {}) or {}
        best = ckpt.get("best", None)
        bad_count = ckpt.get("bad_count", None)
        return epoch, metrics, best, bad_count

    # if ckpt is not dict, it can't resume optimizer/epoch
    raise RuntimeError("Checkpoint is not a dict; cannot resume training state.")


def main():
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, vars(args))

    if cfg["paths"].get("sdlane_root") is None:
        cfg["paths"]["sdlane_root"] = os.environ.get("SDLANE_ROOT")

    if isinstance(cfg["model"].get("pretrained"), str):
        cfg["model"]["pretrained"] = str2bool(cfg["model"]["pretrained"])
    if isinstance(cfg["train"].get("amp"), str):
        cfg["train"]["amp"] = str2bool(cfg["train"]["amp"])

    set_seed(int(cfg["project"]["seed"]))

    sdlane_root = resolve_sdlane_root(cfg)
    run_dir = make_run_dir(cfg)
    dump_yaml(run_dir / "resolved_config.yaml", cfg)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # WandB init (optional)
    # -----------------------------
    wb_cfg = cfg.get("wandb", {}) or {}
    use_wandb = bool(wb_cfg.get("enabled", False))
    if use_wandb:
        import wandb  # lazy import

        wandb.init(
            project=wb_cfg.get("project", "lane-segmentation"),
            entity=wb_cfg.get("entity"),
            name=cfg["paths"].get("run_name"),
            dir=str(run_dir),
            config=cfg,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")


    train_loader, val_loader, _, _ = build_loaders(cfg, sdlane_root, run_dir)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)

    if use_wandb and wb_cfg.get("watch", False):
        wandb.watch(model, log="gradients", log_freq=100)

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
            opt,
            step_size=int(sch_cfg.get("step_size", 10)),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )

    logger = CSVLogger(
        run_dir / "metrics.csv",
        fieldnames=["epoch", "train_loss", "val_loss", "dice", "iou", "precision", "recall", "f1", "lr"],
    )

    # -----------------------------
    # Early Stopping config
    # -----------------------------
    es_cfg = cfg["train"].get("early_stop", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 15))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    warmup_epochs = int(es_cfg.get("warmup_epochs", 0))

    metric_name = cfg["train"]["checkpoint"]["metric"]
    mode = (cfg["train"]["checkpoint"]["mode"] or "max").lower()
    thr = float(cfg["eval"]["threshold"])

    # default best/bad_count
    best = -1e9 if mode == "max" else 1e9
    bad_count = 0
    start_epoch = 1

    def is_improved(current, best_val):
        if mode == "max":
            return current > (best_val + min_delta)
        return current < (best_val - min_delta)

    # -----------------------------
    # Resume (optional)
    # -----------------------------
    resume_path = cfg["train"].get("resume", None)
    # allow nested cfg.train.resume.path style too
    if isinstance(resume_path, dict):
        resume_path = resume_path.get("path")

    if resume_path:
        ckpt_path = Path(resume_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (run_dir / ckpt_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")

        last_epoch, last_metrics, ckpt_best, ckpt_bad = load_checkpoint(
            ckpt_path, model, optimizer=opt, device=device
        )

        start_epoch = int(last_epoch) + 1 if last_epoch is not None else 1

        # best/bad_count 복원(있으면)
        if ckpt_best is not None:
            try:
                best = float(ckpt_best)
            except Exception:
                pass
        if ckpt_bad is not None:
            try:
                bad_count = int(ckpt_bad)
            except Exception:
                pass

        print(f"🔁 Resumed from: {ckpt_path}")
        print(f"   start_epoch={start_epoch}, best={best}, bad_count={bad_count}")
        if use_wandb:
            wandb.log({"resume/ckpt": str(ckpt_path), "resume/start_epoch": start_epoch}, step=start_epoch - 1)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
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

        log_dict = {
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

        logger.log(log_dict)
        if use_wandb:
            wandb.log(log_dict, step=epoch)

        save_every = int(cfg["train"]["checkpoint"].get("save_every", 1))
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model,
                opt,
                epoch,
                metrics,
                cfg,
            )

        current = float(metrics[metric_name])
        improved = is_improved(current, best)
        if improved:
            best = current
            bad_count = 0
            # best 저장할 때 early-stop 상태도 같이 저장되도록 (checkpoint 함수가 dict 저장이면 아래 정보 포함 권장)
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, opt, epoch, metrics, cfg)
        else:
            if es_enabled and epoch > warmup_epochs:
                bad_count += 1

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
            f"best_{metric_name}={best:.4f} "
            f"bad={bad_count}/{patience}"
        )

        if es_enabled and epoch > warmup_epochs and bad_count >= patience:
            print(
                f"🛑 EarlyStopping triggered: no improvement in '{metric_name}' "
                f"for {patience} epochs (min_delta={min_delta}, mode={mode})."
            )
            break

    if use_wandb:
        import wandb
        wandb.finish()

    print(f"✅ Done. outputs: {run_dir}")


if __name__ == "__main__":
    main()
