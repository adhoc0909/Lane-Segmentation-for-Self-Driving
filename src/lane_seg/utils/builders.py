import torch
from torch.utils.data import DataLoader
from lane_seg.data.split import make_splits
from lane_seg.data.transforms import build_train_transforms, build_val_transforms
from lane_seg.data.dataset import SDLaneDataset

def build_loaders(cfg, sdlane_root, run_dir):
    splits_dir = run_dir / "splits"
    train_list = sdlane_root / "train_list.txt"
    train_txt, val_txt = make_splits(train_list, splits_dir, float(cfg["data"]["val_ratio"]), int(cfg["project"]["seed"]))

    train_ds = SDLaneDataset(sdlane_root, train_txt, cfg, transforms=build_train_transforms(cfg))
    val_ds = SDLaneDataset(sdlane_root, val_txt, cfg, transforms=build_val_transforms(cfg))

    dl_args = dict(
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
        persistent_workers=bool(cfg["data"].get("persistent_workers", True)) and int(cfg["data"]["num_workers"]) > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_args)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **dl_args)
    return train_loader, val_loader, train_txt, val_txt
