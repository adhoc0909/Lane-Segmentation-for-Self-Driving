from pathlib import Path
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SDLaneDataset(Dataset):
    def __init__(self, sdlane_root: Path, list_file: Path, cfg, transforms=None):
        self.root = Path(sdlane_root)
        self.list_file = Path(list_file)
        self.cfg = cfg
        self.transforms = transforms
        self.thickness = int(cfg["data"].get("lane_thickness", 6))

        with open(self.list_file, "r", encoding="utf-8") as f:
            self.items = [l.strip() for l in f.readlines() if l.strip()]

    def __len__(self):
        return len(self.items)

    def _parse_item(self, item: str):
        item = item.strip().replace('\\', '/')
        if item.lower().endswith('.jpg') or item.lower().endswith('.png'):
            item = item.rsplit('.', 1)[0]
        parts = [p for p in item.split('/') if p]
        if len(parts) < 2:
            raise ValueError(f'Unknown list format: {item}')
        return parts[-2], parts[-1]

    def _label_to_mask(self, label_path: Path, h: int, w: int):
        mask = np.zeros((h, w), dtype=np.uint8)
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lanes = data.get("lanes") or data.get("Lane") or data.get("lane") or []
        for lane in lanes:
            pts = np.array(lane, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            cv2.polylines(mask, [pts], False, 1, thickness=self.thickness)
        return mask

    def __getitem__(self, idx):
        scene, frame = self._parse_item(self.items[idx])
        img_path = self.root / "images" / scene / f"{frame}.jpg"
        lbl_path = self.root / "labels" / scene / f"{frame}.json"

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]
        mask = self._label_to_mask(lbl_path, h0, w0)

        if self.transforms is not None:
            out = self.transforms(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask
