import torch
from tqdm import tqdm
from lane_seg.evaluation.metrics import dice_from_logits, iou_from_logits

def train_one_epoch(model, loader, optimizer, loss_fn, device, amp=True, grad_clip_norm=0.0):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    losses = []
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.detach().item())
    return float(sum(losses) / max(1, len(losses)))

@torch.no_grad()
def validate(model, loader, loss_fn, device, thr=0.5):
    model.eval()
    losses, dices, ious = [], [], []
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(loss.item())
        dices.append(dice_from_logits(logits, y, thr=thr).mean().item())
        ious.append(iou_from_logits(logits, y, thr=thr).mean().item())
    return {
        "val_loss": float(sum(losses) / max(1, len(losses))),
        "dice": float(sum(dices) / max(1, len(dices))),
        "iou": float(sum(ious) / max(1, len(ious))),
    }
