#!/usr/bin/env python3
"""Export DINOv2 ViT-S/14 patch tokens to a static-size ONNX file.

The nav-stack vision worker letterboxes RGB to a square and expects output
``(1, N, D)`` L2-normalizable patch tokens (no CLS), with
``N = (size / 14) ** 2``.

Examples:

  python tools/export_dinov2_onnx.py
  python tools/export_dinov2_onnx.py --size 448 --out src/semantic/models/dinov2_vits14_patches_448.onnx

Requires: torch (and a network fetch of facebookresearch/dinov2 weights on
first run). onnxruntime is optional (used to smoke-test the export).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=448, help="square input side (multiple of 14)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .onnx path (default: src/semantic/models/dinov2_vits14_patches_<size>.onnx)",
    )
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    size = int(args.size)
    if size < 14 or size % 14 != 0:
        print(f"--size must be a positive multiple of 14, got {size}", file=sys.stderr)
        return 2

    import torch
    import torch.nn as nn

    print(f"loading dinov2_vits14 via torch.hub (size={size})…")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    backbone.eval()

    class PatchTokens(nn.Module):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # (B, N, D) patch tokens only — matches VisionWorker._dino_tokens.
            return self.model.forward_features(x)["x_norm_patchtokens"]

    wrapped = PatchTokens(backbone)
    wrapped.eval()

    root = Path(__file__).resolve().parents[1]
    out = args.out or (
        root / "src" / "semantic" / "models" / f"dinov2_vits14_patches_{size}.onnx"
    )
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, size, size, dtype=torch.float32)
    print(f"exporting → {out}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(out),
            input_names=["images"],
            output_names=["patch_tokens"],
            opset_version=int(args.opset),
            do_constant_folding=True,
            dynamo=False,
        )

    # Quick shape check.
    with torch.no_grad():
        ref = wrapped(dummy)
    expected_n = (size // 14) ** 2
    print(f"torch reference shape: {tuple(ref.shape)} (expect N={expected_n}, D=384)")
    if ref.shape[1] != expected_n:
        print("WARNING: unexpected patch count", file=sys.stderr)

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"images": dummy.numpy()})[0]
        print(f"onnxruntime smoke shape: {got.shape}")
        import numpy as np

        err = float(np.max(np.abs(got - ref.numpy())))
        print(f"max abs err vs torch: {err:.5f}")
    except ImportError:
        print("onnxruntime not installed; skipped smoke test")

    print(f"done: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(
        "Point SlamConfig.vision_dinov2_onnx at this file and omit "
        "vision_synthetic (or set false)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
