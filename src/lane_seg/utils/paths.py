from pathlib import Path
import os

def resolve_sdlane_root(cfg):
    root = cfg["paths"].get("sdlane_root") or os.environ.get("SDLANE_ROOT")
    if not root:
        raise ValueError("Set paths.sdlane_root in config or SDLANE_ROOT env var.")
    p = Path(root).expanduser().resolve()
    if not (p / "train_list.txt").exists():
        raise FileNotFoundError(f"train_list.txt not found under: {p}")
    return p

def make_run_dir(cfg):
    out_dir = Path(cfg["paths"]["out_dir"]).expanduser().resolve()
    run_name = cfg["paths"].get("run_name") or "run"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
