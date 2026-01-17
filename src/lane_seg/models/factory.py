import segmentation_models_pytorch as smp

def build_model(cfg):
    m = cfg["model"]
    arch = m["arch"].lower()
    encoder_weights = "imagenet" if m.get("pretrained", True) else None

    common = dict(
        encoder_name=m["encoder"],
        encoder_weights=encoder_weights,
        in_channels=m.get("in_channels", 3),
        classes=m.get("num_classes", 1),
    )

    if arch == "unet":
        return smp.Unet(**common)
    if arch in ("unetpp", "unetplusplus", "unet++"):
        return smp.UnetPlusPlus(**common)
    if arch in ("deeplabv3plus", "deeplabv3+", "dlv3p"):
        return smp.DeepLabV3Plus(**common)
    if arch == "fpn":
        return smp.FPN(**common)
    raise ValueError(f"Unknown model arch: {arch}")
