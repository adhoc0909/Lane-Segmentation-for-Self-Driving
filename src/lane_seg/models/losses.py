import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        dice = (2*inter + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()

def build_loss(cfg):
    name = cfg["loss"]["name"].lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss(smooth=cfg["loss"].get("smooth", 1.0))
    if name == "bce_dice":
        bce_w = float(cfg["loss"].get("bce_weight", 0.5))
        dice_w = float(cfg["loss"].get("dice_weight", 0.5))
        bce = nn.BCEWithLogitsLoss()
        dice = DiceLoss(smooth=cfg["loss"].get("smooth", 1.0))
        class Combo(nn.Module):
            def forward(self, logits, targets):
                return bce_w*bce(logits, targets) + dice_w*dice(logits, targets)
        return Combo()
    raise ValueError(f"Unknown loss: {name}")
