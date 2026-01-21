from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from lane_seg.yolo.convert import prepare_yolo_seg_dataset


@dataclass
class YoloTrainResult:
    run_dir: Path
    best_weights: Optional[Path]
    last_weights: Optional[Path]


def train_yolo_seg(cfg: Dict[str, Any], sdlane_root: Path, run_dir: Path) -> YoloTrainResult:
    """Train YOLO11-seg using Ultralytics on a dataset converted from SDLane masks."""
    ycfg = cfg.get("yolo", {}) or {}

    # Lazy import to keep base dependencies light
    from ultralytics import YOLO

    ds = prepare_yolo_seg_dataset(cfg, sdlane_root=sdlane_root, run_dir=run_dir)

    weights = str(ycfg.get("weights", "yolo11n-seg.pt"))
    model = YOLO(weights)

    imgsz = int(ycfg.get("imgsz", cfg["data"]["image_size"][0]))
    epochs = int(ycfg.get("epochs", cfg.get("train", {}).get("epochs", 50)))
    batch = int(ycfg.get("batch", cfg.get("train", {}).get("batch_size", 8)))
    device = ycfg.get("device", 0)

    # Keep YOLO-native aug simple; advanced weather/clahe are handled by dataset conversion + your U-Net pipeline
    # You can tune these in cfg['yolo']['hyp']
    hyp = ycfg.get("hyp", {}) or {}

    results = model.train(
        data=str(ds.data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        project=str(run_dir),
        name="yolo",
        exist_ok=True,
        verbose=True,
        **hyp,
    )

    # Ultralytics saves under <project>/<name>/weights/{best,last}.pt
    yolo_run = Path(results.save_dir) if hasattr(results, "save_dir") else (run_dir / "yolo")
    best = yolo_run / "weights" / "best.pt"
    last = yolo_run / "weights" / "last.pt"

    return YoloTrainResult(run_dir=yolo_run, best_weights=best if best.exists() else None, last_weights=last if last.exists() else None)
