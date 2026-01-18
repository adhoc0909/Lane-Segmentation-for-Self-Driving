from __future__ import annotations

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

        data_cfg = cfg.get("data", {})
        self.thickness = int(data_cfg.get("lane_thickness", 6))

        # -----------------------------
        # Precomputed masks 옵션
        # -----------------------------
        self.use_precomputed_masks = bool(data_cfg.get("use_precomputed_masks", False))
        self.mask_subdir = str(data_cfg.get("mask_subdir", "masks"))  # default: <root>/masks/scene/frame.png
        self.mask_ext = str(data_cfg.get("mask_ext", "png")).lstrip(".")
        self.fallback_to_json = bool(data_cfg.get("fallback_to_json", True))

        with open(self.list_file, "r", encoding="utf-8") as f:
            self.items = [l.strip() for l in f.readlines() if l.strip()]

    def __len__(self):
        return len(self.items)

    def _parse_item(self, item: str):
        item = item.strip().replace("\\", "/")
        if item.lower().endswith(".jpg") or item.lower().endswith(".png"):
            item = item.rsplit(".", 1)[0]
        parts = [p for p in item.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"Unknown list format: {item}")
        return parts[-2], parts[-1]

    def _label_to_mask(self, label_path: Path, h: int, w: int) -> np.ndarray:
        """
        JSON label -> binary mask (0/1)
        Supports:
          - SDLane: {"geometry": [ [[x,y],...], ...], "idx":[...] }
          - fallback: "lanes" / "Lane" / "lane"
        """
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lanes = data.get("geometry")
        if lanes is None:
            lanes = data.get("lanes") or data.get("Lane") or data.get("lane") or []

        if not isinstance(lanes, list):
            return mask

        for lane in lanes:
            if not isinstance(lane, list) or len(lane) < 2:
                continue

            pts = np.asarray(lane, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
                continue

            # clip coords to image bounds (coords may be negative / out-of-range)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            pts = pts.astype(np.int32)

            cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=self.thickness)

        return mask

    def _load_precomputed_mask(self, mask_path: Path, h: int, w: int) -> np.ndarray:
        """
        precomputed PNG -> binary mask (0/1)
        """
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # 안전: 사이즈가 다르면 리사이즈(최근접)
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        m = (m > 0).astype(np.uint8)
        return m

    def __getitem__(self, idx):
        scene, frame = self._parse_item(self.items[idx])

        img_path = self.root / "images" / scene / f"{frame}.jpg"
        lbl_path = self.root / "labels" / scene / f"{frame}.json"
        mask_path = self.root / self.mask_subdir / scene / f"{frame}.{self.mask_ext}"

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        # -----------------------------
        # mask 로드 우선순위
        # -----------------------------
        if self.use_precomputed_masks:
            try:
                mask = self._load_precomputed_mask(mask_path, h0, w0)
            except Exception:
                if not self.fallback_to_json:
                    raise
                mask = self._label_to_mask(lbl_path, h0, w0)
        else:
            mask = self._label_to_mask(lbl_path, h0, w0)

        if self.transforms is not None:
            out = self.transforms(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        # if idx < 20:
        #     print("mask_exists:", mask_path.exists(), "mask_path:", mask_path)
        return img, mask
