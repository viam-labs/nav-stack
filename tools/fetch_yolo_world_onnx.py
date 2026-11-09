#!/usr/bin/env python3
"""Download a pre-exported indoor YOLO-Worldv2 ONNX (or fall back to export).

Default destination: ``src/semantic/models/yolov8s-worldv2-indoor.onnx``.
Class prompts live in the sibling ``.classes.json`` (shipped in-module).

  python tools/fetch_yolo_world_onnx.py
  python tools/fetch_yolo_world_onnx.py --export   # force local ultralytics export

Env override: ``NAV_STACK_YOLO_WORLD_URL``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "yolov8s-worldv2-indoor.onnx"
# Attached to the GitHub prerelease that first shipped this asset.
DEFAULT_URL = os.environ.get(
    "NAV_STACK_YOLO_WORLD_URL",
    "https://github.com/viam-labs/nav-stack/releases/download/v1.2.0-rc29/"
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
    ap.add_argument(
        "--export",
        action="store_true",
        help="skip download; run export_yolo_world_onnx.py",
    )
    ap.add_argument("--size", choices=("n", "s", "m", "l", "x"), default="s")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    out: Path = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # yolov8s-world ONNX is typically ~25–40 MB.
    if out.is_file() and out.stat().st_size > 5_000_000:
        print(f"already present: {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0

    if not args.export:
        print(f"downloading {args.url}")
        try:
            urllib.request.urlretrieve(args.url, out)
            if out.stat().st_size < 5_000_000:
                raise RuntimeError(f"download too small ({out.stat().st_size} B)")
            print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"download failed ({exc}); falling back to local export…")
            if out.is_file():
                out.unlink()

    export = ROOT / "tools" / "export_yolo_world_onnx.py"
    classes = out.with_suffix(".classes.json")
    cmd = [
        sys.executable,
        str(export),
        "--size",
        str(args.size),
        "--imgsz",
        str(args.imgsz),
        "--out",
        str(out),
    ]
    if classes.is_file():
        cmd.extend(["--classes", str(classes)])
    print("running:", " ".join(cmd))
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
