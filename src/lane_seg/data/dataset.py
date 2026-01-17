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

    def _label_to_mask(self, json_path, h, w):
        import json
        import cv2
        import numpy as np

        with open(json_path, "r") as f:
            data = json.load(f)

        # ✅ SDLane format
        lanes = data.get("geometry", [])
        mask = np.zeros((h, w), dtype=np.uint8)

        for lane in lanes:
            if len(lane) < 2:
                continue

            pts = np.array(lane, dtype=np.float32)

            # 좌표 클리핑 (음수 / 이미지 밖 방지)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

            pts = pts.astype(np.int32)
            cv2.polylines(
                mask,
                [pts],
                isClosed=False,
                color=1,
                thickness=self.thickness,
            )
        

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
