from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConvGRUCell(nn.Module):
    """Minimal ConvGRU cell for 2D feature maps.

    h_t = (1 - z) * h_{t-1} + z * n
    n = tanh(Wx * x + Wh * (r * h_{t-1}))
    """

    def __init__(self, in_ch: int, hid_ch: int, k: int = 3):
        super().__init__()
        pad = k // 2
        self.in_ch = int(in_ch)
        self.hid_ch = int(hid_ch)

        self.conv_zr = nn.Conv2d(in_ch + hid_ch, 2 * hid_ch, kernel_size=k, padding=pad, bias=True)
        self.conv_n = nn.Conv2d(in_ch + hid_ch, hid_ch, kernel_size=k, padding=pad, bias=True)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor]) -> torch.Tensor:
        if h_prev is None:
            # initialize hidden with zeros (same spatial size as x)
            b, _, h, w = x.shape
            h_prev = torch.zeros((b, self.hid_ch, h, w), device=x.device, dtype=x.dtype)

        cat = torch.cat([x, h_prev], dim=1)
        zr = torch.sigmoid(self.conv_zr(cat))
        z, r = torch.split(zr, self.hid_ch, dim=1)
        cat_n = torch.cat([x, r * h_prev], dim=1)
        n = torch.tanh(self.conv_n(cat_n))
        h = (1.0 - z) * h_prev + z * n
        return h


class RecurrentUNetGRU(nn.Module):
    """UNet with a ConvGRU at the bottleneck.

    Input:
      - x: [B, T, C, H, W] (sequence)
    Output:
      - logits_last: [B, classes, H, W] (for the last frame)

    Notes:
      - Skip connections are taken from the *current* frame (per time step).
      - Only the bottleneck feature is temporally smoothed via ConvGRU.
      - This is designed to be a drop-in replacement for single-frame training in this repo.
    """

    def __init__(
        self,
        encoder_name: str,
        encoder_weights: Optional[str],
        in_channels: int = 3,
        classes: int = 1,
        seq_len: int = 3,
        hidden_mul: float = 1.0,
    ):
        super().__init__()

        # Build a standard SMP UNet but we will manually run its components.
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        # encoder out channels: list for each stage; bottleneck is the last.
        enc_chs: List[int] = list(self.unet.encoder.out_channels)
        bottleneck_ch = int(enc_chs[-1])

        hid_ch = max(8, int(round(bottleneck_ch * float(hidden_mul))))
        self.gru = ConvGRUCell(in_ch=bottleneck_ch, hid_ch=hid_ch, k=3)
        self.hid_to_bottleneck = nn.Conv2d(hid_ch, bottleneck_ch, kernel_size=1, bias=True)

        self.seq_len = int(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backward-compatible: allow single frame [B,C,H,W]
        if x.ndim == 4:
            return self.unet(x)

        if x.ndim != 5:
            raise ValueError(f"Expected x as [B,T,C,H,W] or [B,C,H,W], got shape={tuple(x.shape)}")

        b, t, c, h, w = x.shape
        # We don't force t==self.seq_len (allow dynamic), but typical training uses fixed.

        h_state: Optional[torch.Tensor] = None
        logits_last: Optional[torch.Tensor] = None

        for ti in range(t):
            xt = x[:, ti]
            feats = self.unet.encoder(xt)

            bottleneck = feats[-1]
            h_state = self.gru(bottleneck, h_state)
            bottleneck_refined = self.hid_to_bottleneck(h_state)

            feats = list(feats)
            feats[-1] = bottleneck_refined

            dec = self.unet.decoder(feats)
            logits = self.unet.segmentation_head(dec)
            logits_last = logits


        assert logits_last is not None
        return logits_last
