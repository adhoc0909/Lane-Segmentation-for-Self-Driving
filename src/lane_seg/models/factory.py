import segmentation_models_pytorch as smp


def _is_recurrent_arch(arch: str) -> bool:
    a = (arch or "").lower()
    return a in ("unet_gru", "unet_convlstm", "unet_lstm")

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
    if _is_recurrent_arch(arch):
        # Recurrent UNet (ConvGRU bottleneck). Sequence length is handled by dataset + model forward.
        from lane_seg.models.recurrent_unet_gru import RecurrentUNetGRU  # lazy import

        seq_len = int(cfg.get("data", {}).get("sequence_len", 3))
        hidden_mul = float(m.get("gru_hidden_mul", 1.0))
        return RecurrentUNetGRU(
            encoder_name=common["encoder_name"],
            encoder_weights=common["encoder_weights"],
            in_channels=common["in_channels"],
            classes=common["classes"],
            seq_len=seq_len,
            hidden_mul=hidden_mul,
        )
    if arch in ("unetpp", "unetplusplus", "unet++"):
        return smp.UnetPlusPlus(**common)
    if arch in ("deeplabv3plus", "deeplabv3+", "dlv3p"):
        return smp.DeepLabV3Plus(**common)
    if arch == "fpn":
        return smp.FPN(**common)
    raise ValueError(f"Unknown model arch: {arch}")
