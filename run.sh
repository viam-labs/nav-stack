#!/usr/bin/env bash
# Entrypoint for the nav-stack module. Activates the Python venv and starts the
# module server (builtin SLAM + nav only).
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate

# Module upgrades may skip first_run; ensure vision runtime is present.
if ! python -c "import onnxruntime" >/dev/null 2>&1; then
    echo "[nav-stack] installing missing vision deps…"
    if ! pip install -r requirements-vision.txt -q 2>/dev/null; then
        pip install "onnxruntime>=1.17.0" -q || true
    fi
fi
# Pull vision ONNX weights once if the defaults are absent (non-fatal).
DINOV2_ONNX="src/semantic/models/dinov2_vits14_patches_448.onnx"
if [ ! -f "$DINOV2_ONNX" ] && [ -f tools/fetch_dinov2_onnx.py ]; then
    echo "[nav-stack] fetching DINOv2 ONNX…"
    python tools/fetch_dinov2_onnx.py || true
fi
YOLO_WORLD_ONNX="src/semantic/models/yolov8s-worldv2-indoor.onnx"
if [ ! -f "$YOLO_WORLD_ONNX" ] && [ -f tools/fetch_yolo_world_onnx.py ]; then
    echo "[nav-stack] fetching YOLO-World ONNX…"
    python tools/fetch_yolo_world_onnx.py || true
fi

# The module's numpy work is many small matrix ops on the SLAM / nav threads.
# Pip numpy ships OpenBLAS, which by default spins one worker per core and
# busy-waits between calls; that showed up as hundreds of percent CPU while
# the robot sat still. One BLAS thread per call is faster here, and the
# Python threads still run in parallel across cores.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# glibc gives each thread its own malloc arena; with ~10 threads churning
# multi-MB grids that fragments into a large resident set. Two arenas keep
# RSS bounded with no measurable cost at this allocation rate.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

exec python -m src.main "$@"
