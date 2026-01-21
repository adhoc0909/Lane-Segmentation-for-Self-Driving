"""YOLO backend utilities.

We train Ultralytics YOLO11 instance segmentation on a dataset converted from
binary lane masks. At inference time, we take the union of instance masks to
recover a single semantic lane mask, so we can reuse semantic metrics (Dice/IoU).
"""

from .train import train_yolo_seg, YoloTrainResult
from .eval import eval_yolo_seg, YoloEvalResult
from .convert import prepare_yolo_seg_dataset, YoloSegDatasetPaths
