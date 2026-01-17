import albumentations as A

def build_train_transforms(cfg):
    W, H = cfg["data"]["image_size"]
    aug = cfg.get("augment", {})
    ops = [A.Resize(height=H, width=W)]
    if aug.get("enabled", True):
        ops += [
            A.HorizontalFlip(p=aug.get("hflip_p", 0.0)),
            A.RandomBrightnessContrast(p=aug.get("brightness_contrast_p", 0.2)),
            A.RandomGamma(p=aug.get("gamma_p", 0.1)),
            A.GaussianBlur(p=aug.get("blur_p", 0.05)),
        ]
    return A.Compose(ops)

def build_val_transforms(cfg):
    W, H = cfg["data"]["image_size"]
    return A.Compose([A.Resize(height=H, width=W)])
