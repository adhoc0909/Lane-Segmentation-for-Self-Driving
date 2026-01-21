from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from lane_seg.data.split import make_splits
from lane_seg.data.transforms import (
    build_train_transforms,
    build_val_transforms,
    build_train_transforms_seq,
    build_val_transforms_seq,
)
from lane_seg.data.dataset import SDLaneDataset, SDLaneSequenceDataset


def _use_sequence_dataset(cfg) -> bool:
    seq_len = int((cfg.get("data", {}) or {}).get("sequence_len", 1) or 1)
    arch = str(((cfg.get("model", {}) or {}).get("arch", "") or "")).lower()
    return seq_len > 1 or arch in ("unet_gru", "unet_convlstm", "unet_lstm")


def _get_split_cfg(cfg, split: str) -> Tuple[str, str]:
    """
    Returns (split_dir, list_name)

    Priority:
      1) cfg["data"]["splits"][split]["dir"/"list"]
      2) default: dir=split, list=f"{split}_list.txt"
    """
    data = cfg.get("data", {}) or {}
    splits = data.get("splits", {}) or {}
    s = splits.get(split, {}) or {}
    split_dir = str(s.get("dir", split))
    list_name = str(s.get("list", f"{split}_list.txt"))
    return split_dir, list_name


def _resolve_list_path(sdlane_root: Path, split_dir: str, list_name: str, split: str) -> Path:
    """
    Find list file without hardcoding.

    Candidates (in order):
      A) <root>/<split_dir>/<list_name>   (e.g., root/test/test_list.txt)
      B) <root>/<list_name>              (e.g., root/test_list.txt)
      C) <root>/<split>_list.txt         (legacy)
    """
    p1 = sdlane_root / split_dir / list_name
    if p1.exists():
        return p1

    p2 = sdlane_root / list_name
    if p2.exists():
        return p2

    p3 = sdlane_root / f"{split}_list.txt"
    return p3


def build_loaders(cfg, sdlane_root: Path, run_dir: Path, test: bool = False):
    """
    Returns:
      - if test=False: (train_loader, val_loader, train_txt, val_txt)
      - if test=True : (test_loader, test_txt)

    Supports dataset structures:
      1) root/train/images|labels|masks + root/train/train_list.txt
      2) root/images|labels|masks + root/train_list.txt (legacy)
      3) root/test/images|labels|masks + root/test/test_list.txt (your current)
    """
    sdlane_root = Path(sdlane_root)

    dl_args = dict(
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        persistent_workers=bool(cfg["data"].get("persistent_workers", True)) and int(cfg["data"]["num_workers"]) > 0,
    )

    # -----------------------------
    # TEST ONLY (no splitting)
    # -----------------------------
    if test:
        test_split_dir, test_list_name = _get_split_cfg(cfg, "test")
        test_txt = _resolve_list_path(sdlane_root, test_split_dir, test_list_name, "test")
        if not test_txt.exists():
            raise FileNotFoundError(f"test list not found. tried: {test_txt}")

        if _use_sequence_dataset(cfg):
            seq_len = int((cfg.get("data", {}) or {}).get("sequence_len", 3) or 3)
            frame_step = int((cfg.get("data", {}) or {}).get("frame_step", 1) or 1)
            test_ds = SDLaneSequenceDataset(
                sdlane_root,
                test_txt,
                cfg,
                transforms=build_val_transforms_seq(cfg, seq_len),
                split_dir=test_split_dir,
                sequence_len=seq_len,
                frame_step=frame_step,
            )
        else:
            test_ds = SDLaneDataset(
                sdlane_root,
                test_txt,
                cfg,
                transforms=build_val_transforms(cfg),
                split_dir=test_split_dir,
            )
        test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **dl_args)
        return test_loader, test_txt

    # -----------------------------
    # TRAIN/VAL
    # -----------------------------
    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_split_dir, train_list_name = _get_split_cfg(cfg, "train")
    train_list = _resolve_list_path(sdlane_root, train_split_dir, train_list_name, "train")
    if not train_list.exists():
        raise FileNotFoundError(f"train list not found. tried: {train_list}")

    train_txt, val_txt = make_splits(
        train_list,
        splits_dir,
        float(cfg["data"]["val_ratio"]),
        int(cfg["project"]["seed"]),
    )

    if _use_sequence_dataset(cfg):
        seq_len = int((cfg.get("data", {}) or {}).get("sequence_len", 3) or 3)
        frame_step = int((cfg.get("data", {}) or {}).get("frame_step", 1) or 1)
        train_ds = SDLaneSequenceDataset(
            sdlane_root,
            train_txt,
            cfg,
            transforms=build_train_transforms_seq(cfg, seq_len),
            split_dir=train_split_dir,
            sequence_len=seq_len,
            frame_step=frame_step,
        )
        val_ds = SDLaneSequenceDataset(
            sdlane_root,
            val_txt,
            cfg,
            transforms=build_val_transforms_seq(cfg, seq_len),
            split_dir=train_split_dir,
            sequence_len=seq_len,
            frame_step=frame_step,
        )
    else:
        train_ds = SDLaneDataset(
            sdlane_root,
            train_txt,
            cfg,
            transforms=build_train_transforms(cfg),
            split_dir=train_split_dir,
        )
        val_ds = SDLaneDataset(
            sdlane_root,
            val_txt,
            cfg,
            transforms=build_val_transforms(cfg),
            split_dir=train_split_dir,
        )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_args)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **dl_args)
    return train_loader, val_loader, train_txt, val_txt
