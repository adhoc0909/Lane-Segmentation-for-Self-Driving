from pathlib import Path
import random

def read_list(list_path: Path):
    with open(list_path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]

def make_splits(train_list_path: Path, out_dir: Path, val_ratio: float, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    if train_txt.exists() and val_txt.exists():
        return train_txt, val_txt

    items = read_list(train_list_path)
    rnd = random.Random(seed)
    rnd.shuffle(items)

    n_val = int(len(items) * val_ratio)
    val_items = items[:n_val]
    train_items = items[n_val:]

    train_txt.write_text("\n".join(train_items) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_items) + "\n", encoding="utf-8")
    return train_txt, val_txt
