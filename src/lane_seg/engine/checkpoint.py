from pathlib import Path
import torch

def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "cfg": cfg,
    }, path)

def load_weights(path: Path, model):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
