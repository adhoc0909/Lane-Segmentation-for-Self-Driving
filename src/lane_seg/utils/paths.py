from pathlib import Path
import os


def resolve_sdlane_root(cfg):
    root = cfg["paths"].get("sdlane_root") or os.environ.get("SDLANE_ROOT")
    if not root:
        raise ValueError("Set paths.sdlane_root in config or SDLANE_ROOT env var.")

    p = Path(root).expanduser().resolve()

    # ✅ 허용 구조:
    # 1) <root>/train_list.txt or <root>/test_list.txt (legacy)
    # 2) <root>/train/train_list.txt or <root>/test/test_list.txt (split folders)
    has_root_lists = (p / "train_list.txt").exists() or (p / "test_list.txt").exists()
    has_split_lists = (p / "train" / "train_list.txt").exists() or (p / "test" / "test_list.txt").exists()

    if not has_root_lists and not has_split_lists:
        raise FileNotFoundError(
            f"List file not found under: {p}\n"
            f"Expected one of:\n"
            f"  - {p/'train_list.txt'}\n"
            f"  - {p/'test_list.txt'}\n"
            f"  - {p/'train'/'train_list.txt'}\n"
            f"  - {p/'test'/'test_list.txt'}"
        )

    return p


def make_run_dir(cfg):
    out_dir = Path(cfg["paths"]["out_dir"]).expanduser().resolve()
    run_name = cfg["paths"].get("run_name") or "run"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
