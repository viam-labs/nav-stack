#!/usr/bin/env python3
"""Export YOLO-Worldv2 ONNX with indoor landmark class prompts baked in.

Robots normally get this file via ``setup.sh`` / ``tools/fetch_yolo_world_onnx.py``
(GitHub release asset). Use this script only to rebuild the asset.

Ultralytics YOLO-World lets you set open-vocab classes before export; the
resulting ONNX has one score channel per prompt (class id = list index).
Class prompts come from ``src/semantic/models/yolov8s-worldv2-indoor.classes.json``.

Examples:

  python tools/export_yolo_world_onnx.py
  python tools/export_yolo_world_onnx.py --size s --imgsz 640 \\
      --out src/semantic/models/yolov8s-worldv2-indoor.onnx

Requires: ``pip install ultralytics`` (torch + network on first run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES_PATH = (
    ROOT / "src" / "semantic" / "models" / "yolov8s-worldv2-indoor.classes.json"
)


def _default_classes() -> list[str]:
    if DEFAULT_CLASSES_PATH.is_file():
        raw = json.loads(DEFAULT_CLASSES_PATH.read_text())
        if isinstance(raw, list) and all(isinstance(c, str) for c in raw):
            return list(raw)
    raise SystemExit(f"missing default class list: {DEFAULT_CLASSES_PATH}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--size",
        choices=("n", "s", "m", "l", "x"),
        default="s",
        help="YOLOv8-Worldv2 width (s ≈ fast local CPU/OpenVINO)",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output .onnx path",
    )
    ap.add_argument(
        "--classes",
        type=Path,
        default=None,
        help="optional JSON list of class prompt strings",
    )
    ap.add_argument(
        "--classes-json-out",
        type=Path,
        default=None,
        help="write the baked class list next to the ONNX (for config copy-paste)",
    )
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is required: pip install ultralytics",
            file=sys.stderr,
        )
        return 2

    if args.classes is not None:
        classes = json.loads(Path(args.classes).read_text())
        if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes):
            print("--classes must be a JSON list of strings", file=sys.stderr)
            return 2
    else:
        classes = _default_classes()

    out = args.out
    if out is None:
        out = (
            ROOT
            / "src"
            / "semantic"
            / "models"
            / f"yolov8{args.size}-worldv2-indoor.onnx"
        )
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    weights = f"yolov8{args.size}-worldv2.pt"
    print(f"loading {weights}…")
    model = YOLO(weights)
    print(f"set_classes ({len(classes)}): {classes}")
    model.set_classes(classes)
    # Warmup forward pass so text embeddings are fused before ONNX export.
    try:
        import numpy as np

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy, conf=0.01, verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"warmup predict skipped ({exc})")

    print(f"exporting onnx imgsz={args.imgsz} → {out}")
    exported = model.export(format="onnx", imgsz=int(args.imgsz), simplify=True)
    exported_path = Path(str(exported)).resolve()
    if exported_path != out:
        out.write_bytes(exported_path.read_bytes())
        # Ultralytics often writes next to the .pt; leave that copy, prefer ours.
        print(f"copied {exported_path} → {out}")

    classes_out = args.classes_json_out
    if classes_out is None:
        classes_out = out.with_suffix(".classes.json")
    classes_out.write_text(json.dumps(classes, indent=2) + "\n")
    print(f"wrote class list → {classes_out}")
    print(
        "\nConfig snippet:\n"
        f'  "vision_yolo_onnx": "{out.name}",\n'
        f'  "vision_detector_classes": {json.dumps(classes)},\n'
        '  "vision_prefer_detections": true\n'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
