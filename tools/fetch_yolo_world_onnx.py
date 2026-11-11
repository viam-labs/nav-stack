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
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "yolov8s-worldv2-indoor.onnx"
# Rebuilt with the person prompt. Older copies (v1.2.0-rc29) cannot see people.
DEFAULT_URL = os.environ.get(
    "NAV_STACK_YOLO_WORLD_URL",
    "https://github.com/viam-labs/nav-stack/releases/download/v1.2.0-rc49/"
    + DEFAULT_NAME,
)
_MIN_ONNX_BYTES = 5_000_000


def class_list_rev(classes_path: Path) -> str:
    """Short hash of the baked class list. A new prompt changes it."""
    if not classes_path.is_file():
        return ""
    return hashlib.sha256(classes_path.read_bytes()).hexdigest()[:16]


def stamp_path(onnx_path: Path) -> Path:
    return onnx_path.with_name(onnx_path.name + ".stamp")


def onnx_is_current(onnx_path: Path, classes_path: Path) -> bool:
    """True when ``onnx_path`` was fetched for this class list."""
    if not onnx_path.is_file() or onnx_path.stat().st_size <= _MIN_ONNX_BYTES:
        return False
    expected = class_list_rev(classes_path)
    if not expected:
        return True
    stamp = stamp_path(onnx_path)
    return stamp.is_file() and stamp.read_text().strip() == expected


def write_stamp(onnx_path: Path, classes_path: Path) -> None:
    expected = class_list_rev(classes_path)
    if expected:
        stamp_path(onnx_path).write_text(expected + "\n")


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
    classes = out.with_suffix(".classes.json")
    # yolov8s-world ONNX is typically ~25–40 MB. A file left over from an
    # older class list (no person prompt) is replaced.
    if not args.export and onnx_is_current(out, classes):
        print(f"already present: {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0

    if not args.export:
        print(f"downloading {args.url}")
        partial = out.with_suffix(".onnx.partial")
        try:
            urllib.request.urlretrieve(args.url, partial)
            if partial.stat().st_size < _MIN_ONNX_BYTES:
                raise RuntimeError(f"download too small ({partial.stat().st_size} B)")
            partial.replace(out)
            write_stamp(out, classes)
            print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
            return 0
        except Exception as exc:  # noqa: BLE001
            if partial.is_file():
                partial.unlink()
            if out.is_file() and out.stat().st_size > _MIN_ONNX_BYTES:
                print(
                    f"download failed ({exc}); keeping existing {out.name}",
                    file=sys.stderr,
                )
                return 1
            print(f"download failed ({exc}); falling back to local export…")
            if out.is_file():
                out.unlink()

    export = ROOT / "tools" / "export_yolo_world_onnx.py"
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
    code = int(subprocess.call(cmd))
    if code == 0 and out.is_file() and not onnx_is_current(out, classes):
        write_stamp(out, classes)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
