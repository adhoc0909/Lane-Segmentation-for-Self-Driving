from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# -----------------------------
# bootstrap: repo imports
# -----------------------------
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]
SRC_DIR = REPO_ROOT / "src"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args():
    p = argparse.ArgumentParser("Export various models to ONNX and save alongside.")
    p.add_argument("--in", dest="in_path", required=True, help="Input model path (.pt/.pth/.onnx)")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (default: same dir as input)")
    p.add_argument("--name", type=str, default=None, help="Output onnx filename (default: input stem + .onnx)")
    p.add_argument("--opset", type=int, default=12)

    # common input shape
    p.add_argument("--h", type=int, default=288)
    p.add_argument("--w", type=int, default=512)
    p.add_argument("--c", type=int, default=3)
    p.add_argument("--batch", type=int, default=1)

    # device for export tracing
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    # backend override (optional)
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "ultralytics", "lane_seg", "mmseg", "torchscript", "onnx"],
                   help="Force backend instead of auto-detect.")

    # lane_seg (your repo) needs config to rebuild model
    p.add_argument("--config", type=str, default="configs/default.yaml", help="Config for lane_seg backend")

    # mmseg/mmcv (optional): needs config + checkpoint
    p.add_argument("--mmseg_config", type=str, default=None)
    p.add_argument("--mmseg_ckpt", type=str, default=None)

    # export options
    p.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for H/W (and batch)")
    return p.parse_args()


def _resolve_out_path(in_path: Path, out_dir: Optional[str], name: Optional[str]) -> Path:
    out_root = Path(out_dir) if out_dir else in_path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    out_name = name if name else (in_path.stem + ".onnx")
    return out_root / out_name


def _auto_backend(in_path: Path) -> str:
    suf = in_path.suffix.lower()
    if suf == ".onnx":
        return "onnx"
    # torchscript is still .pt generally; we detect later by loading
    # default: try ultralytics first (common for .pt), else lane_seg torch
    return "auto"


def _to_device(device_str: str) -> torch.device:
    if device_str.lower().startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dummy_input(batch: int, c: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    x = torch.randn(batch, c, h, w, device=device, dtype=torch.float32)
    return x


def _dummy_input_seq(batch: int, t: int, c: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Sequence dummy input: [B,T,C,H,W]."""
    x = torch.randn(batch, t, c, h, w, device=device, dtype=torch.float32)
    return x


def _export_torch_model_to_onnx(
    model: torch.nn.Module,
    out_path: Path,
    dummy: torch.Tensor,
    opset: int,
    dynamic: bool,
) -> None:
    model.eval()

    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = None
    if dynamic:
        if dummy.ndim == 4:
            dynamic_axes = {
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }
        elif dummy.ndim == 5:
            # input: [B,T,C,H,W], output: [B,classes,H,W]
            dynamic_axes = {
                "input": {0: "batch", 1: "time", 3: "height", 4: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }
        else:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def _try_load_as_torchscript(in_path: Path, device: torch.device) -> Optional[torch.jit.RecursiveScriptModule]:
    try:
        m = torch.jit.load(str(in_path), map_location=device)
        return m
    except Exception:
        return None


def export_auto(in_path: Path, out_path: Path, args) -> Tuple[str, Path]:
    """
    Returns (backend_used, out_path)
    """
    device = _to_device(args.device)
    dummy = _dummy_input(args.batch, args.c, args.h, args.w, device)

    # 0) already onnx
    if in_path.suffix.lower() == ".onnx" or args.backend == "onnx":
        if in_path.resolve() != out_path.resolve():
            shutil.copy2(in_path, out_path)
        return "onnx(copy)", out_path

    # 1) torchscript?
    if args.backend in ("auto", "torchscript"):
        ts = _try_load_as_torchscript(in_path, device)
        if ts is not None:
            # TorchScript -> ONNX export is possible but not always reliable.
            # For "unified runtime" goal, prefer exporting from original eager model.
            # Here we do best-effort ONNX export from TorchScript module.
            _export_torch_model_to_onnx(ts, out_path, dummy, args.opset, args.dynamic)
            return "torchscript", out_path
        if args.backend == "torchscript":
            raise RuntimeError("backend=torchscript was forced but input is not a TorchScript module.")

    # 2) ultralytics
    if args.backend in ("auto", "ultralytics"):
        try:
            from ultralytics import YOLO  # type: ignore
            yolo = YOLO(str(in_path))
            # ultralytics will create output next to weights unless path is given
            # We export to a temp path then move to out_path to guarantee location/name.
            tmp_dir = out_path.parent / "_tmp_ultra_export"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            exported = yolo.export(format="onnx", opset=args.opset)  # returns path-like in many versions
            exported_path = Path(exported) if isinstance(exported, (str, Path)) else None

            if exported_path is None or not exported_path.exists():
                # fallback: search in_path.stem + .onnx around input dir
                cand = in_path.with_suffix(".onnx")
                if cand.exists():
                    exported_path = cand
                else:
                    # search common locations
                    found = list(in_path.parent.glob(in_path.stem + "*.onnx"))
                    if found:
                        exported_path = found[0]

            if exported_path is None or not exported_path.exists():
                raise RuntimeError("Ultralytics export did not produce an ONNX file.")

            if exported_path.resolve() != out_path.resolve():
                shutil.copy2(exported_path, out_path)

            # cleanup tmp dir (best-effort)
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

            return "ultralytics", out_path

        except Exception as e:
            if args.backend == "ultralytics":
                raise
            # auto fallback to lane_seg / torch
            pass

    # 3) mmseg/mmcv
    if args.backend in ("mmseg",):
        if not args.mmseg_config or not args.mmseg_ckpt:
            raise RuntimeError("mmseg backend requires --mmseg_config and --mmseg_ckpt.")
        cfg_path = Path(args.mmseg_config)
        ckpt_path = Path(args.mmseg_ckpt)
        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError("mmseg_config or mmseg_ckpt not found.")

        try:
            # mmseg v1.x style
            from mmseg.apis import init_model  # type: ignore
            from mmseg.utils import register_all_modules  # type: ignore
            register_all_modules()

            model = init_model(str(cfg_path), str(ckpt_path), device=str(device))
            # mmseg model may return logits with shape [B, C, H, W] (C=classes)
            # ONNX export: best-effort
            _export_torch_model_to_onnx(model, out_path, dummy, args.opset, args.dynamic)
            return "mmseg", out_path
        except Exception as e:
            raise RuntimeError(f"mmseg export failed: {e}") from e

    # 4) lane_seg (your repo): rebuild model via config and load weights
    if args.backend in ("auto", "lane_seg"):
        from lane_seg.utils.config import load_yaml  # noqa: E402
        from lane_seg.models.factory import build_model  # noqa: E402
        from lane_seg.engine.checkpoint import load_weights  # noqa: E402

        cfg = load_yaml(args.config)
        model = build_model(cfg).to(device).eval()

        # If the config indicates a sequence-based model, export with 5D input [B,T,C,H,W].
        seq_len = int((cfg.get("data", {}) or {}).get("sequence_len", 1) or 1)
        arch = str(((cfg.get("model", {}) or {}).get("arch", "") or "")).lower()
        if seq_len > 1 or arch in ("unet_gru", "unet_convlstm", "unet_lstm"):
            dummy_in = _dummy_input_seq(args.batch, seq_len, args.c, args.h, args.w, device)
        else:
            dummy_in = dummy

        # Try to load weights; if it's a pure state_dict without known keys,
        # load_weights should handle common formats in your repo.
        load_weights(in_path, model)

        _export_torch_model_to_onnx(model, out_path, dummy_in, args.opset, args.dynamic)
        return "lane_seg", out_path

    raise RuntimeError("Unsupported input for export. Try --backend ultralytics/lane_seg/mmseg/torchscript explicitly.")


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_path = _resolve_out_path(in_path, args.out_dir, args.name)

    backend = args.backend if args.backend != "auto" else _auto_backend(in_path)
    args.backend = backend

    used, saved = export_auto(in_path, out_path, args)
    print(f"✅ Export done. backend={used}")
    print(f"✅ Saved: {saved}")


if __name__ == "__main__":
    main()
