#!/usr/bin/env python3
"""Download a pre-exported DINOv2 patch-token ONNX (or fall back to export).

Default destination: ``src/semantic/models/dinov2_vits14_patches_448.onnx``
(next to the module sources — same path the robot config should point at).

  python tools/fetch_dinov2_onnx.py
  python tools/fetch_dinov2_onnx.py --url https://…/dinov2_vits14_patches_448.onnx
  python tools/fetch_dinov2_onnx.py --export   # force local torch export

Env override: ``NAV_STACK_DINOV2_URL``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "dinov2_vits14_patches_448.onnx"
# Filled in when we attach the asset to the GitHub release; override with --url.
DEFAULT_URL = os.environ.get(
    "NAV_STACK_DINOV2_URL",
    "https://github.com/viam-labs/nav-stack/releases/download/v1.2.0-rc3/"
    + DEFAULT_NAME,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "src" / "semantic" / "models" / DEFAULT_NAME,
    )
    ap.add_argument("--url", type=str, default=DEFAULT_URL)
    ap.add_argument("--export", action="store_true", help="skip download; run export_dinov2_onnx.py")
    ap.add_argument("--size", type=int, default=448)
    args = ap.parse_args()

    out: Path = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 1_000_000:
        print(f"already present: {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0

    if not args.export:
        print(f"downloading {args.url}")
        try:
            urllib.request.urlretrieve(args.url, out)
            if out.stat().st_size < 1_000_000:
                raise RuntimeError(f"download too small ({out.stat().st_size} B)")
            print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"download failed ({exc}); falling back to local export…")
            if out.is_file():
                out.unlink()

    export = ROOT / "tools" / "export_dinov2_onnx.py"
    cmd = [sys.executable, str(export), "--size", str(args.size), "--out", str(out)]
    print("running:", " ".join(cmd))
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
