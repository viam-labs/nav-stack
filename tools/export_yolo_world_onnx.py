#!/usr/bin/env python3
"""Export YOLO-Worldv2 ONNX with indoor landmark class prompts baked in.

Ultralytics YOLO-World lets you set open-vocab classes before export; the
resulting ONNX has one score channel per prompt (class id = list index).
Point nav-stack ``vision_yolo_onnx`` at the file and set
``vision_detector_classes`` to the same list (order matters).

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

# Keep in sync with SlamConfig.vision_landmark_labels defaults / README.
DEFAULT_CLASSES = [
    "painting",
    "poster",
    "picture frame",
    "artwork",
    "tv",
    "monitor",
    "door",
    "fire extinguisher",
    "clock",
    "whiteboard",
    "sign",
    "exit sign",
    "potted plant",
    "bookshelf",
    "cabinet",
    "laptop",
    "book",
    "chair",
    "couch",
    "bed",
    "dining table",
    "refrigerator",
    "sink",
    "bottle",
    "vase",
]


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
        classes = list(DEFAULT_CLASSES)

    root = Path(__file__).resolve().parents[1]
    out = args.out
    if out is None:
        out = (
            root
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
