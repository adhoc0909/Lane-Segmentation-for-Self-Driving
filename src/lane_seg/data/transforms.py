import albumentations as A
from albumentations.pytorch import ToTensorV2


def _maybe_add(ops, op):
    """Append an augmentation op if not None."""
    if op is not None:
        ops.append(op)


def _imgnet_normalize_ops():
    """Standard ImageNet normalization for pretrained encoders."""
    return [
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]


def build_train_transforms(cfg):
    """Albumentations transforms for training.

    Notes:
      - All geometric transforms are mask-synced.
      - Normalize is applied AFTER all augmentations.
      - Some ops (RandomFog/RandomRain) may be absent depending on Albumentations version;
        we guard with hasattr.
    """
    W, H = cfg["data"]["image_size"]  # [W, H]
    aug = cfg.get("augment", {}) or {}

    ops = [A.Resize(height=H, width=W)]

    if aug.get("enabled", True):
        # -----------------------------
        # Geometric (mask-synced)
        # -----------------------------
        hflip_p = float(aug.get("hflip_p", 0.0))
        if hflip_p > 0:
            ops.append(A.HorizontalFlip(p=hflip_p))

        ssr_p = float(aug.get("shift_scale_rotate_p", 0.0))
        if ssr_p > 0:
            ops.append(
                A.ShiftScaleRotate(
                    shift_limit=float(aug.get("shift_limit", 0.02)),
                    scale_limit=float(aug.get("scale_limit", 0.05)),
                    rotate_limit=float(aug.get("rotate_limit", 3.0)),
                    border_mode=0,  # cv2.BORDER_CONSTANT
                    value=0,
                    mask_value=0,
                    p=ssr_p,
                )
            )

        # -----------------------------
        # Photometric
        # -----------------------------
        clahe_p = float(aug.get("clahe_p", 0.0))
        _maybe_add(
            ops,
            A.CLAHE(
                clip_limit=float(aug.get("clahe_clip_limit", 2.0)),
                tile_grid_size=tuple(aug.get("clahe_tile_grid_size", [8, 8])),
                p=clahe_p,
            )
            if clahe_p > 0
            else None,
        )

        bc_p = float(aug.get("brightness_contrast_p", 0.0))
        _maybe_add(
            ops,
            A.RandomBrightnessContrast(
                brightness_limit=float(aug.get("brightness_limit", 0.2)),
                contrast_limit=float(aug.get("contrast_limit", 0.2)),
                p=bc_p,
            )
            if bc_p > 0
            else None,
        )

        gamma_p = float(aug.get("gamma_p", 0.0))
        _maybe_add(
            ops,
            A.RandomGamma(
                gamma_limit=tuple(aug.get("gamma_limit", [80, 120])),
                p=gamma_p,
            )
            if gamma_p > 0
            else None,
        )

        hsv_p = float(aug.get("hsv_p", 0.0))
        _maybe_add(
            ops,
            A.HueSaturationValue(
                hue_shift_limit=int(aug.get("hue_shift_limit", 10)),
                sat_shift_limit=int(aug.get("sat_shift_limit", 15)),
                val_shift_limit=int(aug.get("val_shift_limit", 10)),
                p=hsv_p,
            )
            if hsv_p > 0
            else None,
        )

        rgb_p = float(aug.get("rgb_shift_p", 0.0))
        _maybe_add(
            ops,
            A.RGBShift(
                r_shift_limit=int(aug.get("r_shift_limit", 10)),
                g_shift_limit=int(aug.get("g_shift_limit", 10)),
                b_shift_limit=int(aug.get("b_shift_limit", 10)),
                p=rgb_p,
            )
            if rgb_p > 0
            else None,
        )

        # -----------------------------
        # Blur / Noise / Compression (light)
        # -----------------------------
        gb_p = float(aug.get("gaussian_blur_p", 0.0))
        _maybe_add(
            ops,
            A.GaussianBlur(
                blur_limit=tuple(aug.get("gaussian_blur_limit", [3, 5])),
                p=gb_p,
            )
            if gb_p > 0
            else None,
        )

        mb_p = float(aug.get("motion_blur_p", 0.0))
        _maybe_add(
            ops,
            A.MotionBlur(
                blur_limit=int(aug.get("motion_blur_limit", 5)),
                p=mb_p,
            )
            if mb_p > 0
            else None,
        )

        gn_p = float(aug.get("gauss_noise_p", 0.0))
        _maybe_add(
            ops,
            A.GaussNoise(
                var_limit=tuple(aug.get("gauss_noise_var_limit", [10.0, 50.0])),
                mean=float(aug.get("gauss_noise_mean", 0.0)),
                p=gn_p,
            )
            if gn_p > 0
            else None,
        )

        jpeg_p = float(aug.get("jpeg_p", 0.0))
        _maybe_add(
            ops,
            A.ImageCompression(
                quality_lower=int(aug.get("jpeg_quality_lower", 70)),
                quality_upper=int(aug.get("jpeg_quality_upper", 100)),
                p=jpeg_p,
            )
            if jpeg_p > 0
            else None,
        )

        # Backward-compatible alias
        blur_p = float(aug.get("blur_p", 0.0))
        _maybe_add(ops, A.GaussianBlur(p=blur_p) if blur_p > 0 else None)

        # -----------------------------
        # Weather (low probability)
        # -----------------------------
        rain_p = float(aug.get("rain_p", 0.0))
        if rain_p > 0 and hasattr(A, "RandomRain"):
            ops.append(
                A.RandomRain(
                    drop_length=int(aug.get("rain_drop_length", 20)),
                    drop_width=int(aug.get("rain_drop_width", 1)),
                    blur_value=int(aug.get("rain_blur_value", 3)),
                    brightness_coefficient=float(aug.get("rain_brightness_coeff", 0.9)),
                    p=rain_p,
                )
            )

        fog_p = float(aug.get("fog_p", 0.0))
        if fog_p > 0 and hasattr(A, "RandomFog"):
            ops.append(
                A.RandomFog(
                    fog_coef_lower=float(aug.get("fog_coef_lower", 0.1)),
                    fog_coef_upper=float(aug.get("fog_coef_upper", 0.3)),
                    alpha_coef=float(aug.get("fog_alpha_coef", 0.08)),
                    p=fog_p,
                )
            )

    # Normalize + tensor (always)
    ops += _imgnet_normalize_ops()
    return A.Compose(ops)


def build_val_transforms(cfg):
    W, H = cfg["data"]["image_size"]
    ops = [A.Resize(height=H, width=W)]
    ops += _imgnet_normalize_ops()
    return A.Compose(ops)
