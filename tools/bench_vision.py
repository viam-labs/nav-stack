#!/usr/bin/env python3
"""Phase 0 vision latency bench (Meerkat / NUC).

Runs the VisionWorker on synthetic RGB-D frames (and optionally real ONNX
weights). Reports p50/p95 inference latency and whether OpenVINO is available.

Examples:

  python tools/bench_vision.py
  python tools/bench_vision.py --frames 100 --short-side 224
  python tools/bench_vision.py --dinov2 /path/to/dinov2_vits14.onnx

Pass criteria (from the plan, synthetic path is a smoke test only):
  - Capture-to-result p95 <= 250 ms on the synthetic or ONNX path
  - With OpenVINO GPU + real DINOv2: p95 <= 40 ms preferred
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.semantic.onnx_session import available_providers, openvino_gpu_available
from src.semantic.patches import backproject_patches
from src.semantic.types import CameraIntrinsics, VisionFrame
from src.semantic.worker import VisionWorker, WorkerConfig


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def _make_frame(h: int = 480, w: int = 640) -> VisionFrame:
    # Textured color so synthetic descriptors are not all identical.
    yy, xx = np.mgrid[0:h, 0:w]
    color = np.stack(
        [
            (xx % 64) * 4,
            (yy % 48) * 5,
            ((xx + yy) % 80) * 3,
        ],
        axis=-1,
    ).astype(np.uint8)
    # Flat wall at 2 m with a few holes.
    depth = np.full((h, w), 2.0, dtype=np.float32)
    depth[100:120, 200:220] = 0.0
    K = CameraIntrinsics(
        fx=0.5 * w,
        fy=0.5 * h,
        cx=0.5 * w,
        cy=0.5 * h,
        width=w,
        height=h,
    )
    return VisionFrame(
        camera_name="bench-cam",
        stamp=time.time(),
        captured_at=time.monotonic(),
        color_rgb=color,
        depth_m=depth,
        intrinsics=K,
        mount_x=0.15,
        mount_z=0.4,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--short-side", type=int, default=224)
    ap.add_argument("--dinov2", type=str, default="")
    ap.add_argument("--yolo", type=str, default="")
    ap.add_argument("--cpu-threads", type=int, default=2)
    args = ap.parse_args()

    providers = available_providers()
    print(f"onnxruntime providers: {providers or '(onnxruntime not installed)'}")
    print(f"OpenVINO GPU available: {openvino_gpu_available()}")

    synthetic = not bool(args.dinov2)
    cfg = WorkerConfig(
        dinov2_onnx=args.dinov2,
        yolo_onnx=args.yolo,
        input_short_side=args.short_side,
        cpu_threads=args.cpu_threads,
        synthetic=synthetic,
    )
    worker = VisionWorker(cfg)
    print(f"worker provider: {worker.stats['provider']}  synthetic={cfg.synthetic}")

    frame = _make_frame()
    times_ms: list[float] = []
    backproj_ms: list[float] = []
    n_valid = 0

    for i in range(args.warmup + args.frames):
        frame.captured_at = time.monotonic()
        t0 = time.perf_counter()
        result = worker.process(frame)
        t1 = time.perf_counter()
        obs = backproject_patches(frame, result.patches)
        t2 = time.perf_counter()
        if i < args.warmup:
            continue
        times_ms.append((t1 - t0) * 1000.0)
        backproj_ms.append((t2 - t1) * 1000.0)
        n_valid = len(obs)

    worker.close()
    if not times_ms:
        print("no frames measured")
        return 2

    p50 = _percentile(times_ms, 50)
    p95 = _percentile(times_ms, 95)
    mean = statistics.fmean(times_ms)
    print(
        f"infer_ms: mean={mean:.1f}  p50={p50:.1f}  p95={p95:.1f}  "
        f"n={len(times_ms)}  patches_valid≈{n_valid}"
    )
    print(
        f"backproject_ms: p50={_percentile(backproj_ms, 50):.2f}  "
        f"p95={_percentile(backproj_ms, 95):.2f}"
    )

    # Soft gates for CI / laptop; Meerkat ONNX gates are documented in the plan.
    ok = p95 <= 250.0
    print(f"gate capture-to-infer p95<=250ms: {'PASS' if ok else 'FAIL'} ({p95:.1f})")
    if args.dinov2 and openvino_gpu_available():
        gpu_ok = p95 <= 40.0
        print(
            f"gate OpenVINO DINOv2 p95<=40ms: "
            f"{'PASS' if gpu_ok else 'FAIL'} ({p95:.1f})"
        )
        ok = ok and gpu_ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
